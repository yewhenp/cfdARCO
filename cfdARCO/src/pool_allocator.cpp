#include "pool_allocator.hpp"

std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> Allocator::cuda_mem_pool = {nullptr};
bool Allocator::allocator_alive = false;