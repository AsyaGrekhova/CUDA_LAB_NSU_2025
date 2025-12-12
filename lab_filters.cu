// lab_filters.cu
// GPU image filters: Gaussian blur, Sobel edge, Median denoise
// Two GPU variants: shared-memory kernels and texture-object kernels
// Uses libpng for I/O. Compile: nvcc -O3 -arch=sm_70 -o lab_filters lab_filters.cu -lpng

#include <cuda_runtime.h>
#include <png.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <unistd.h>
#include <limits.h>
#include <cstring>

#define CHECK_CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    std::cerr<<"CUDA error "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(EXIT_FAILURE);} } while(0)

// Simple RGBA image container
struct Image {
    int width=0, height=0, channels=4;
    std::vector<unsigned char> data;
    void resize(int w, int h){ width=w; height=h; channels=4; data.assign(w*h*4,0); }
};

// Helper: get directory of current executable (Linux)
static std::string get_exe_dir() {
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len == -1) {
        // fallback to current working directory
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd))) return std::string(cwd);
        return std::string();
    }
    buf[len] = '\0';
    // find last slash
    char *slash = strrchr(buf, '/');
    if (!slash) return std::string(buf);
    *slash = '\0';
    return std::string(buf);
}

// ---------------- PNG I/O (libpng) - minimal robust reader/writer ----------------
bool read_png(const std::string &fname, Image &img){
    FILE* fp=fopen(fname.c_str(),"rb"); if(!fp) return false;
    png_structp png=png_create_read_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);
    if(!png){ fclose(fp); return false; }
    png_infop info=png_create_info_struct(png);
    if(!info){ png_destroy_read_struct(&png,nullptr,nullptr); fclose(fp); return false; }
    if(setjmp(png_jmpbuf(png))){ png_destroy_read_struct(&png,&info,nullptr); fclose(fp); return false; }
    png_init_io(png,fp); png_read_info(png, info);
    png_uint_32 w=png_get_image_width(png,info), h=png_get_image_height(png,info);
    png_byte color=png_get_color_type(png,info), depth=png_get_bit_depth(png,info);
    if(depth==16) png_set_strip_16(png);
    if(color==PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if(color==PNG_COLOR_TYPE_GRAY && depth<8) png_set_expand_gray_1_2_4_to_8(png);
    if(png_get_valid(png,info,PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if(color==PNG_COLOR_TYPE_RGB || color==PNG_COLOR_TYPE_GRAY || color==PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png,0xFF,PNG_FILLER_AFTER);
    if(color==PNG_COLOR_TYPE_GRAY || color==PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_read_update_info(png, info);
    img.width=w; img.height=h; img.channels=4; img.data.assign(w*h*4,0);
    std::vector<png_bytep> rows(h);
    for (int y=0;y<h;y++) rows[y] = (png_bytep)&img.data[y*w*4];
    png_read_image(png, rows.data());
    png_read_end(png,nullptr);
    png_destroy_read_struct(&png,&info,nullptr);
    fclose(fp);
    return true;
}
bool write_png(const std::string &fname, const Image &img){
    FILE* fp=fopen(fname.c_str(),"wb"); if(!fp) return false;
    png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);
    if(!png){ fclose(fp); return false; }
    png_infop info=png_create_info_struct(png);
    if(!info){ png_destroy_write_struct(&png,nullptr); fclose(fp); return false; }
    if(setjmp(png_jmpbuf(png))){ png_destroy_write_struct(&png,&info); fclose(fp); return false; }
    png_init_io(png,fp);
    png_set_IHDR(png, info, img.width, img.height, 8, PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<png_bytep> rows(img.height);
    for (int y=0;y<img.height;y++) rows[y] = const_cast<png_bytep>(&img.data[y*img.width*4]);
    png_write_image(png, rows.data()); png_write_end(png,nullptr);
    png_destroy_write_struct(&png,&info); fclose(fp);
    return true;
}

// ---------------- CPU reference (Gaussian 3x3) ----------------
void gaussian_cpu(const Image &in, Image &out){
    out.resize(in.width,in.height);
    const float k[3][3]={{1.f/16,2.f/16,1.f/16},{2.f/16,4.f/16,2.f/16},{1.f/16,2.f/16,1.f/16}};
    for(int y=0;y<in.height;y++) for(int x=0;x<in.width;x++){
        for(int c=0;c<4;c++){
            float sum=0.f;
            for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
                int nx = std::min(std::max(x+dx,0), in.width-1);
                int ny = std::min(std::max(y+dy,0), in.height-1);
                sum += in.data[(ny*in.width+nx)*4 + c] * k[dy+1][dx+1];
            }
            out.data[(y*in.width+x)*4 + c] = (unsigned char) std::round(std::min(255.f, std::max(0.f, sum)));
        }
    }
}

// ---------------- GPU: shared-memory kernels ----------------
__device__ inline uchar4 read_uc4(const unsigned char* base, int idx){
    uchar4 v; v.x = base[idx*4+0]; v.y = base[idx*4+1]; v.z = base[idx*4+2]; v.w = base[idx*4+3]; return v;
}
__device__ inline void write_uc4(unsigned char* base, int idx, const uchar4 &v){
    base[idx*4+0] = v.x; base[idx*4+1] = v.y; base[idx*4+2] = v.z; base[idx*4+3] = v.w;
}

// Shared-memory Gaussian (block tile + halo)
__global__ void gaussian_shared(const unsigned char* __restrict__ in, unsigned char* out, int w, int h){
    const int R=1;
    extern __shared__ unsigned char s[]; // bytes
    int tx=threadIdx.x, ty=threadIdx.y;
    int bx=blockIdx.x*blockDim.x, by=blockIdx.y*blockDim.y;
    int x=bx+tx, y=by+ty;
    int sW = blockDim.x + 2*R;
    // load into shared (RGBA)
    for(int c=0;c<4;c++){
        int sx = tx + R;
        int sy = ty + R;
        int gx = min(max(x,0), w-1);
        int gy = min(max(y,0), h-1);
        s[(sy*sW + sx)*4 + c] = in[(gy*w + gx)*4 + c];
        if(tx < R){
            int gx2 = min(max(bx + tx - R, 0), w-1);
            s[(sy*sW + (sx-R))*4 + c] = in[(gy*w + gx2)*4 + c];
        }
        if(tx >= blockDim.x - R){
            int gx2 = min(max(bx + tx + R, 0), w-1);
            s[(sy*sW + (sx+R))*4 + c] = in[(gy*w + gx2)*4 + c];
        }
        if(ty < R){
            int gy2 = min(max(by + ty - R, 0), h-1);
            s[((sy-R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c];
        }
        if(ty >= blockDim.y - R){
            int gy2 = min(max(by + ty + R, 0), h-1);
            s[((sy+R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c];
        }
    }
    __syncthreads();
    if(x>=w||y>=h) return;
    const float k[3][3] = {{1.f/16,2.f/16,1.f/16},{2.f/16,4.f/16,2.f/16},{1.f/16,2.f/16,1.f/16}};
    for(int c=0;c<4;c++){
        float sum=0.f;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int sx = tx + dx + R;
            int sy = ty + dy + R;
            sum += (float) s[(sy*sW + sx)*4 + c] * k[dy+1][dx+1];
        }
        out[(y*w + x)*4 + c] = (unsigned char) roundf(fminf(255.f, fmaxf(0.f, sum)));
    }
}

// Shared-memory Sobel
__global__ void sobel_shared(const unsigned char* __restrict__ in, unsigned char* out, int w, int h){
    const int R=1;
    extern __shared__ unsigned char s[];
    int tx=threadIdx.x, ty=threadIdx.y;
    int bx=blockIdx.x*blockDim.x, by=blockIdx.y*blockDim.y;
    int x=bx+tx, y=by+ty;
    int sW = blockDim.x + 2*R;
    for(int c=0;c<4;c++){
        int sx=tx+R, sy=ty+R;
        int gx=min(max(x,0),w-1), gy=min(max(y,0),h-1);
        s[(sy*sW + sx)*4 + c] = in[(gy*w + gx)*4 + c];
        if(tx<R){ int gx2=min(max(bx+tx-R,0),w-1); s[(sy*sW + (sx-R))*4 + c] = in[(gy*w + gx2)*4 + c]; }
        if(tx>=blockDim.x-R){ int gx2=min(max(bx+tx+R,0),w-1); s[(sy*sW + (sx+R))*4 + c] = in[(gy*w + gx2)*4 + c]; }
        if(ty<R){ int gy2=min(max(by+ty-R,0),h-1); s[((sy-R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c]; }
        if(ty>=blockDim.y-R){ int gy2=min(max(by+ty+R,0),h-1); s[((sy+R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c]; }
    }
    __syncthreads();
    if(x<1||x>=w-1||y<1||y>=h-1) return;
    const int sxm[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    const int symm[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    for(int c=0;c<4;c++){
        int gxv=0, gyv=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int sxx = tx + dx + R;
            int syy = ty + dy + R;
            int val = s[(syy*sW + sxx)*4 + c];
            gxv += val * sxm[dy+1][dx+1];
            gyv += val * symm[dy+1][dx+1];
        }
        int mag = (int) sqrtf((float)(gxv*gxv + gyv*gyv));
        mag = 255 - min(255,max(0,mag));
        out[(y*w + x)*4 + c] = (unsigned char) mag;
    }
}

// Shared-memory median (3x3)
__global__ void median_shared(const unsigned char* __restrict__ in, unsigned char* out, int w, int h){
    const int R=1;
    extern __shared__ unsigned char s[];
    int tx=threadIdx.x, ty=threadIdx.y;
    int bx=blockIdx.x*blockDim.x, by=blockIdx.y*blockDim.y;
    int x=bx+tx, y=by+ty;
    int sW = blockDim.x + 2*R;
    for(int c=0;c<4;c++){
        int sx=tx+R, sy=ty+R;
        int gx=min(max(x,0),w-1), gy=min(max(y,0),h-1);
        s[(sy*sW + sx)*4 + c] = in[(gy*w + gx)*4 + c];
        if(tx<R){ int gx2=min(max(bx+tx-R,0),w-1); s[(sy*sW + (sx-R))*4 + c] = in[(gy*w + gx2)*4 + c]; }
        if(tx>=blockDim.x-R){ int gx2=min(max(bx+tx+R,0),w-1); s[(sy*sW + (sx+R))*4 + c] = in[(gy*w + gx2)*4 + c]; }
        if(ty<R){ int gy2=min(max(by+ty-R,0),h-1); s[((sy-R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c]; }
        if(ty>=blockDim.y-R){ int gy2=min(max(by+ty+R,0),h-1); s[((sy+R)*sW + sx)*4 + c] = in[(gy2*w + gx)*4 + c]; }
    }
    __syncthreads();
    if(x<1||x>=w-1||y<1||y>=h-1) return;
    for(int c=0;c<4;c++){
        unsigned char window[9];
        int idx=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int sx = tx+dx+R, sy=ty+dy+R;
            window[idx++] = s[(sy*sW + sx)*4 + c];
        }
        for(int i=0;i<8;i++) for(int j=0;j<8-i;j++) if(window[j]>window[j+1]){ unsigned char t=window[j]; window[j]=window[j+1]; window[j+1]=t; }
        out[(y*w + x)*4 + c] = window[4];
    }
}

// ---------------- GPU: texture-based kernels ----------------
__global__ void gaussian_texture(cudaTextureObject_t texObj, unsigned char* out, int w, int h){
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=w||y>=h) return;
    const float k[3][3]={{1.f/16,2.f/16,1.f/16},{2.f/16,4.f/16,2.f/16},{1.f/16,2.f/16,1.f/16}};
    for(int c=0;c<4;c++){
        float sum=0.f;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int sx = min(max(x+dx,0), w-1);
            int sy = min(max(y+dy,0), h-1);
            uchar4 v = tex2D<uchar4>(texObj, sx, sy);
            unsigned char val = (c==0)?v.x:((c==1)?v.y:((c==2)?v.z:v.w));
            sum += val * k[dy+1][dx+1];
        }
        out[(y*w + x)*4 + c] = (unsigned char) roundf( fminf(255.f, fmaxf(0.f, sum)) );
    }
}

__global__ void sobel_texture(cudaTextureObject_t texObj, unsigned char* out, int w, int h){
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    if(x<1||x>=w-1||y<1||y>=h-1) return;
    const int sxm[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    const int sym[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
    for(int c=0;c<4;c++){
        int gx=0, gy=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            uchar4 v = tex2D<uchar4>(texObj, x+dx, y+dy);
            int val = (c==0)?v.x:((c==1)?v.y:((c==2)?v.z:v.w));
            gx += val * sxm[dy+1][dx+1];
            gy += val * sym[dy+1][dx+1];
        }
        int mag = (int) sqrtf((float)(gx*gx + gy*gy));
        mag = 255 - min(255,max(0,mag));
        out[(y*w + x)*4 + c] = (unsigned char) mag;
    }
}

__global__ void median_texture(cudaTextureObject_t texObj, unsigned char* out, int w, int h){
    int x=blockIdx.x*blockDim.x + threadIdx.x;
    int y=blockIdx.y*blockDim.y + threadIdx.y;
    if(x<1||x>=w-1||y<1||y>=h-1) return;
    for(int c=0;c<4;c++){
        unsigned char window[9]; int idx=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            uchar4 v = tex2D<uchar4>(texObj, x+dx, y+dy);
            window[idx++] = (c==0)?v.x:((c==1)?v.y:((c==2)?v.z:v.w));
        }
        for(int i=0;i<8;i++) for(int j=0;j<8-i;j++) if(window[j]>window[j+1]){ unsigned char t=window[j]; window[j]=window[j+1]; window[j+1]=t; }
        out[(y*w + x)*4 + c] = window[4];
    }
}

// ---------------- Host wrappers ----------------
float run_shared_filter(const Image &in, Image &out, const std::string &filter){
    int w=in.width, h=in.height; size_t bytes=(size_t)w*h*4;
    unsigned char *d_in=nullptr, *d_out=nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes)); CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, in.data.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(16,16), grid((w+15)/16,(h+15)/16);
    // shared bytes: (block.x + 2)*(block.y + 2)*4
    size_t sbytes = (size_t)(block.x + 2) * (block.y + 2) * 4;
    cudaEvent_t st, ed; CHECK_CUDA(cudaEventCreate(&st)); CHECK_CUDA(cudaEventCreate(&ed));
    CHECK_CUDA(cudaEventRecord(st));
    if(filter=="blur") gaussian_shared<<<grid,block,sbytes>>>(d_in,d_out,w,h);
    else if(filter=="edge") sobel_shared<<<grid,block,sbytes>>>(d_in,d_out,w,h);
    else if(filter=="denoise") median_shared<<<grid,block,sbytes>>>(d_in,d_out,w,h);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ed)); CHECK_CUDA(cudaEventSynchronize(ed));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,st,ed));
    out.resize(w,h); CHECK_CUDA(cudaMemcpy(out.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out); cudaEventDestroy(st); cudaEventDestroy(ed);
    return ms;
}

float run_texture_filter(const Image &in, Image &out, const std::string &filter){
    int w=in.width, h=in.height; size_t bytes=(size_t)w*h*4;
    // create cudaArray of uchar4 and copy
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cuArr=nullptr;
    CHECK_CUDA(cudaMallocArray(&cuArr, &desc, w, h));
    std::vector<uchar4> tmp(w*h);
    for(int i=0;i<w*h;i++){
        tmp[i].x = in.data[i*4+0];
        tmp[i].y = in.data[i*4+1];
        tmp[i].z = in.data[i*4+2];
        tmp[i].w = in.data[i*4+3];
    }
    CHECK_CUDA(cudaMemcpyToArray(cuArr, 0,0, tmp.data(), bytes, cudaMemcpyHostToDevice));
    // resource & texture objects
    cudaResourceDesc resDesc; memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray; resDesc.res.array.array = cuArr;
    cudaTextureDesc texDesc; memset(&texDesc,0,sizeof(texDesc));
    texDesc.addressMode[0]=cudaAddressModeClamp; texDesc.addressMode[1]=cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint; texDesc.readMode = cudaReadModeElementType; texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj=0; CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    unsigned char *d_out=nullptr; CHECK_CUDA(cudaMalloc(&d_out, bytes));
    dim3 block(16,16), grid((w+15)/16,(h+15)/16);
    cudaEvent_t st, ed; CHECK_CUDA(cudaEventCreate(&st)); CHECK_CUDA(cudaEventCreate(&ed));
    CHECK_CUDA(cudaEventRecord(st));
    if(filter=="blur") gaussian_texture<<<grid,block>>>(texObj,d_out,w,h);
    else if(filter=="edge") sobel_texture<<<grid,block>>>(texObj,d_out,w,h);
    else if(filter=="denoise") median_texture<<<grid,block>>>(texObj,d_out,w,h);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(ed)); CHECK_CUDA(cudaEventSynchronize(ed));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,st,ed));
    out.resize(w,h); CHECK_CUDA(cudaMemcpy(out.data.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    cudaDestroyTextureObject(texObj); cudaFreeArray(cuArr); cudaFree(d_out);
    cudaEventDestroy(st); cudaEventDestroy(ed);
    return ms;
}

// ---------------- Main ----------------
int main(int argc, char** argv){
    // Determine input file:
    // If argv[1] provided -> use it. Else, look for "input.png" in the same dir as executable.
    std::string infile;
    if(argc>1 && argv[1] && std::strlen(argv[1])>0) infile = std::string(argv[1]);
    else {
        std::string dir = get_exe_dir();
        if(dir.empty()) infile = "input.png";
        else infile = dir + "/input.png";
    }

    Image in;
    if(!read_png(infile, in)){
        std::cerr<<"Cannot open "<<infile<<", creating test image.\n";
        in.resize(512,512);
        for(int y=0;y<in.height;y++) for(int x=0;x<in.width;x++){
            int idx=(y*in.width+x)*4;
            in.data[idx] = (unsigned char)((x*255)/in.width);
            in.data[idx+1] = (unsigned char)((y*255)/in.height);
            in.data[idx+2] = (unsigned char)(((x+y)*255)/(in.width+in.height));
            in.data[idx+3] = 255;
            if(rand()%20==0){ in.data[idx]=rand()%256; in.data[idx+1]=rand()%256; in.data[idx+2]=rand()%256; }
        }
        write_png("input.png", in);
    }
    std::cout<<"Image loaded: "<<in.width<<"x"<<in.height<<"\n";

    std::vector<std::string> filters = {"blur","edge","denoise"};
    // CPU reference for blur only
    Image cpu_out; double cpu_ms=0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        gaussian_cpu(in, cpu_out);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        write_png("cpu_blur.png", cpu_out);
    }

    for(auto &f: filters){
        std::cout<<"\n--- Filter: "<<f<<" ---\n";
        Image out_shared, out_tex;
        float t_shared = run_shared_filter(in, out_shared, f);
        write_png(std::string("gpu_shared_")+f+".png", out_shared);
        float t_tex = run_texture_filter(in, out_tex, f);
        write_png(std::string("gpu_texture_")+f+".png", out_tex);
        std::cout<<"GPU shared time (ms): "<<t_shared<<"\n";
        std::cout<<"GPU texture time (ms): "<<t_tex<<"\n";
        if(f=="blur") {
            std::cout<<"CPU time (ms): "<<cpu_ms<<"\n";
            if(t_shared>0) std::cout<<"Speedup (CPU/GPU-shared): "<<(cpu_ms / (double)t_shared)<<"x\n";
            if(t_tex>0) std::cout<<"Speedup (CPU/GPU-text): "<<(cpu_ms / (double)t_tex)<<"x\n";
        }
    }
    std::cout<<"\nOutputs saved: cpu_blur.png, gpu_shared_*.png, gpu_texture_*.png\n";
    return 0;
}