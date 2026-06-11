#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <mooncake_ep_exception.cuh>
#include <torch/torch.h>

namespace mooncake {

using DeviceStream = at::cuda::CUDAStream;
constexpr torch::DeviceType kDeviceType = torch::kCUDA;

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(kDeviceType);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const DeviceStream& stream) {
        event = std::make_shared<torch::Event>(kDeviceType);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
    }
};

inline torch::Event create_event(const DeviceStream& s) {
    auto event = torch::Event(kDeviceType);
    event.record(s);
    return event;
}

inline void stream_wait(const DeviceStream& s_0, const DeviceStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const DeviceStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}  // namespace mooncake
