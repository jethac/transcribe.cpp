#pragma once

static bool ggml_vk_khr_profile_extensions(const vk::PhysicalDevice & physical) {
    const char * required[] = {
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_3_EXTENSION_NAME,
    };
    const auto available = physical.enumerateDeviceExtensionProperties();
    for (const char * name : required) {
        bool found = false;
        for (const auto & extension : available) {
            found |= std::strcmp(extension.extensionName, name) == 0;
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

struct ggml_vk_khr_profile_features {
    VkPhysicalDevice16BitStorageFeaturesKHR storage16{};
    VkPhysicalDevice8BitStorageFeaturesKHR storage8{};
    VkPhysicalDeviceShaderFloat16Int8FeaturesKHR arithmetic{};
    VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timeline{};

    ggml_vk_khr_profile_features() {
        storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
        storage8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
        arithmetic.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
        timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
        storage16.pNext = &storage8;
        storage8.pNext = &arithmetic;
        arithmetic.pNext = &timeline;
    }

    bool required_supported() const {
        return storage16.storageBuffer16BitAccess && storage8.storageBuffer8BitAccess &&
               arithmetic.shaderInt8 && timeline.timelineSemaphore;
    }

    void mirror(VkPhysicalDeviceVulkan11Features & f11,
                VkPhysicalDeviceVulkan12Features & f12) const {
        f11.storageBuffer16BitAccess = storage16.storageBuffer16BitAccess;
        f11.uniformAndStorageBuffer16BitAccess = storage16.uniformAndStorageBuffer16BitAccess;
        f11.storagePushConstant16 = storage16.storagePushConstant16;
        f11.storageInputOutput16 = storage16.storageInputOutput16;
        f12.storageBuffer8BitAccess = storage8.storageBuffer8BitAccess;
        f12.uniformAndStorageBuffer8BitAccess = storage8.uniformAndStorageBuffer8BitAccess;
        f12.storagePushConstant8 = storage8.storagePushConstant8;
        f12.shaderFloat16 = arithmetic.shaderFloat16;
        f12.shaderInt8 = arithmetic.shaderInt8;
        f12.timelineSemaphore = timeline.timelineSemaphore;
    }

    static void append_extensions(std::vector<const char *> & extensions) {
        extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_MAINTENANCE_3_EXTENSION_NAME);
    }
};

static uint32_t ggml_vk_khr_profile_descriptor_capacity(const vk::PhysicalDeviceProperties & props) {
    const uint32_t capacity = std::min<uint32_t>(MAX_PARAMETER_COUNT,
        std::min(props.limits.maxPerStageDescriptorStorageBuffers,
                 props.limits.maxDescriptorSetStorageBuffers));
    if (capacity < 5) {
        throw std::runtime_error("KHR profile requires at least five storage-buffer descriptors");
    }
    return capacity;
}
