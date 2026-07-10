#pragma once

#include "renderer.h"
#include "SDL_compat.h"

#include <atomic>
#include <vector>

class SdlAudioRenderer : public IAudioRenderer
{
public:
    SdlAudioRenderer();

    virtual ~SdlAudioRenderer();

    virtual bool prepareForPlayback(const OPUS_MULTISTREAM_CONFIGURATION* opusConfig) override;

    virtual void* getAudioBuffer(int* size) override;

    virtual bool submitAudio(int bytesWritten) override;

    virtual void notifyAudioDiscontinuity() override;

    virtual AudioFormat getAudioBufferFormat() override;

    virtual void updateMetrics() override;

private:
    SDL_AudioDeviceID m_AudioDevice;
    void* m_AudioBuffer;
    Uint32 m_FrameSize;
    Uint32 m_FrameDurationMs;
    Uint32 m_ChannelCount;
    Uint32 m_FadeFrames;
    int m_JitterBufferMs;
    bool m_HadProducerDrop;
    std::vector<float> m_LastQueuedSamples;
    std::atomic<Uint32> m_DropCount;
    std::atomic<int> m_QueuedAudioSize;
};
