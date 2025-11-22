// g++ -std=c++17 cpp_ring.cpp -o cpp_ring -pthread

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <iostream>
#include <cstring>

static const char* SHM_NAME = "/ring_buffer_demo";
static const char* FD_FILE  = "/dev/shm/ring_buffer_fds";

static const size_t N = 8;
static const size_t ITER = 10;

struct SharedMemory {
    float buffers[2][N];
};

int main() {
    size_t size = sizeof(SharedMemory);

    // Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, size);

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    auto* shm = reinterpret_cast<SharedMemory*>(addr);

    // Create eventfds
    int ev_c2py = eventfd(0, EFD_SEMAPHORE);
    int ev_py2c = eventfd(0, EFD_SEMAPHORE);

    // Store eventfd numbers in a small file so python can read them
    FILE* f = fopen(FD_FILE, "w");
    fprintf(f, "%d %d", ev_c2py, ev_py2c);
    fclose(f);

    // Main loop
    for (size_t it = 0; it < ITER; ++it) {
        size_t b = it % 2;

        // Fill data
        for (size_t i = 0; i < N; ++i)
            shm->buffers[b][i] = i + it * 1000.0f;

        std::cout << "[C++] send iter " << it << " buf " << b << ": ";
        for (size_t i = 0; i < N; ++i) std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";

        // Notify Python
        uint64_t one = 1;
        write(ev_c2py, &one, sizeof(one));

        std::cout << "OK1\n";
        // Wait for Python → C++
        uint64_t val;
        read(ev_py2c, &val, sizeof(val));
        std::cout << "OK2\n";

        std::cout << "[C++] recv iter " << it << " buf " << b << ": ";
        for (size_t i = 0; i < N; ++i) std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";
    }

    munmap(addr, size);
    close(fd);
    shm_unlink(SHM_NAME);
    unlink(FD_FILE);

    return 0;
}


