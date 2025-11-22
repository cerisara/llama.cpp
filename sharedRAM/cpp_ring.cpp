// g++ -std=c++17 cpp_ring.cpp -o cpp_ring -pthread

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <atomic>

static const char* SHM_NAME = "/ring_buffer_demo";
static const size_t N = 8;                 // array size (small for demo)
static const size_t ITER = 10;

// Memory layout
struct SharedMemory {
    std::atomic<int> c_to_py_flag[2];
    std::atomic<int> py_to_c_flag[2];
    float buffers[2][N];
};

int main() {
    size_t size = sizeof(SharedMemory);

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }

    if (ftruncate(fd, size) != 0) { perror("ftruncate"); return 1; }

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) { perror("mmap"); return 1; }

    auto* shm = reinterpret_cast<SharedMemory*>(addr);

    // Initialize flags
    shm->c_to_py_flag[0] = 0;
    shm->c_to_py_flag[1] = 0;
    shm->py_to_c_flag[0] = 0;
    shm->py_to_c_flag[1] = 0;

    for (size_t it = 0; it < ITER; ++it) {
        size_t b = it % 2;

        // Fill data A
        for (size_t i = 0; i < N; ++i)
            shm->buffers[b][i] = (float)i + it * 1000.0f;

        std::cout << "[C++] Iter " << it << " sending buffer " << b << ": ";
        for (size_t i = 0; i < N; ++i)
            std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";

        // Mark C→ Python ready
        shm->c_to_py_flag[b].store(1, std::memory_order_release);

        // Wait for Python
        while (shm->py_to_c_flag[b].load(std::memory_order_acquire) == 0) {
            usleep(1000);
        }

        // Python placed result in the same buffer index (ring)
        std::cout << "[C++] Received back: ";
        for (size_t i = 0; i < N; ++i)
            std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";

        // Reset flag
        shm->py_to_c_flag[b].store(0, std::memory_order_release);
    }

    munmap(addr, size);
    close(fd);
    shm_unlink(SHM_NAME);
    return 0;
}

