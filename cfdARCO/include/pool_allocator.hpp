#ifndef CFDARCO_POOL_ALLOCATOR_HPP
#define CFDARCO_POOL_ALLOCATOR_HPP

#include <memory>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

class Allocator {
public:
    static std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> cuda_mem_pool;
    static bool allocator_alive;
};

#endif //CFDARCO_POOL_ALLOCATOR_HPP
