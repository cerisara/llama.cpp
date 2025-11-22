// g++ -std=c++17 cpp_ring.cpp -o cpp_ring -pthread

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

static const char* SHM_NAME = "/ring_buffer_demo";
static const char* SEM_C2P = "/c2py_sem";
static const char* SEM_P2C = "/py2c_sem";

static const size_t N = 8;
static const size_t ITER = 10;

struct SharedMemory {
    float buffers[2][N];
};

int main() {
    // Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedMemory));
    void* addr = mmap(nullptr, sizeof(SharedMemory),
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    auto* shm = reinterpret_cast<SharedMemory*>(addr);

    // Create semaphores
    sem_t* sem_c2p = sem_open(SEM_C2P, O_CREAT, 0666, 0);
    sem_t* sem_py2c = sem_open(SEM_P2C, O_CREAT, 0666, 0);

    for (size_t it = 0; it < ITER; ++it) {
        size_t b = it % 2;

        // Fill buffer
        for (size_t i = 0; i < N; ++i)
            shm->buffers[b][i] = i + it * 1000.0f;

        std::cout << "[C++] Sending buffer " << b << ": ";
        for (size_t i = 0; i < N; ++i) std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";

        // Notify Python
        sem_post(sem_c2p);

        // Wait for Python to process
        sem_wait(sem_py2c);

        std::cout << "[C++] Received buffer " << b << ": ";
        for (size_t i = 0; i < N; ++i) std::cout << shm->buffers[b][i] << " ";
        std::cout << "\n";
    }

    // Cleanup
    munmap(addr, sizeof(SharedMemory));
    close(fd);
    shm_unlink(SHM_NAME);
    sem_close(sem_c2p);
    sem_close(sem_py2c);
    sem_unlink(SEM_C2P);
    sem_unlink(SEM_P2C);

    return 0;
}

