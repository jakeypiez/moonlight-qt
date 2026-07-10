#include "sdl.h"
#include "imgui/devui.h"
#include "settings/streamingpreferences.h"

#include <Limelight.h>

namespace {

void crossfadeFromLast(float* samples,
                       uint32_t frames,
                       uint32_t channels,
                       uint32_t fadeFrames,
                       const std::vector<float>& lastSamples)
{
    fadeFrames = qMin(frames, fadeFrames);
    if (fadeFrames == 0 || lastSamples.size() < channels) {
        return;
    }

    for (uint32_t frame = 0; frame < fadeFrames; frame++) {
        const float mix = fadeFrames == 1 ? 1.0f : (float)frame / (fadeFrames - 1);
        for (uint32_t channel = 0; channel < channels; channel++) {
            float& sample = samples[frame * channels + channel];
            sample = lastSamples[channel] * (1.0f - mix) + sample * mix;
        }
    }
}

}

SdlAudioRenderer::SdlAudioRenderer()
    : m_AudioDevice(0),
      m_AudioBuffer(nullptr),
      m_FrameSize(0),
      m_FrameDurationMs(0),
      m_ChannelCount(0),
      m_FadeFrames(0),
      m_JitterBufferMs(80),
      m_HadProducerDrop(false),
      m_DropCount(0),
      m_QueuedAudioSize{0}
{
    SDL_assert(!SDL_WasInit(SDL_INIT_AUDIO));

    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "SDL_InitSubSystem(SDL_INIT_AUDIO) failed: %s",
                     SDL_GetError());
        SDL_assert(SDL_WasInit(SDL_INIT_AUDIO));
    }
}

bool SdlAudioRenderer::prepareForPlayback(const OPUS_MULTISTREAM_CONFIGURATION* opusConfig)
{
    SDL_AudioSpec want, have;

    m_JitterBufferMs = qBound(30, StreamingPreferences::get()->audioJitterBufferMs, 150);

    SDL_zero(want);
    want.freq = opusConfig->sampleRate;
    want.format = AUDIO_F32SYS;
    want.channels = opusConfig->channelCount;

    // On PulseAudio systems, setting a value too small can cause underruns for other
    // applications sharing this output device. We impose a floor of 480 samples (10 ms)
    // to mitigate this issue. Otherwise, we will buffer up to 3 frames of audio which
    // is 15 ms at regular 5 ms frames and 30 ms at 10 ms frames for slow connections.
    // The buffering helps avoid audio underruns due to network jitter.
    want.samples = SDL_max(480, opusConfig->samplesPerFrame * 3);

    m_FrameDurationMs = opusConfig->samplesPerFrame / (opusConfig->sampleRate / 1000);
    m_FrameSize = opusConfig->samplesPerFrame *
                  opusConfig->channelCount *
                  getAudioBufferSampleSize();
    m_ChannelCount = opusConfig->channelCount;
    m_FadeFrames = qMax(1, opusConfig->sampleRate / 1000);
    m_HadProducerDrop = false;
    m_LastQueuedSamples.assign(m_ChannelCount, 0.0f);

    m_AudioDevice = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
    if (m_AudioDevice == 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Failed to open audio device: %s",
                     SDL_GetError());
        return false;
    }

    m_AudioBuffer = SDL_malloc(m_FrameSize);
    if (m_AudioBuffer == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Failed to allocate audio buffer");
        return false;
    }

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "Desired audio buffer: %u samples (%u bytes)",
                want.samples,
                want.samples * want.channels * getAudioBufferSampleSize());

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "Obtained audio buffer: %u samples (%u bytes)",
                have.samples,
                have.size);

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "SDL audio driver: %s",
                SDL_GetCurrentAudioDriver());

    // Start playback
    SDL_PauseAudioDevice(m_AudioDevice, 0);

    DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
        metrics.opusChannelCount = have.channels;
        metrics.audioSampleRate = opusConfig->sampleRate;
        metrics.audioFrameDurationMs = m_FrameDurationMs;

    });

    return true;
}

SdlAudioRenderer::~SdlAudioRenderer()
{
    if (m_AudioDevice != 0) {
        // Stop playback
        SDL_PauseAudioDevice(m_AudioDevice, 1);
        SDL_CloseAudioDevice(m_AudioDevice);
    }

    if (m_AudioBuffer != nullptr) {
        SDL_free(m_AudioBuffer);
    }

    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    SDL_assert(!SDL_WasInit(SDL_INIT_AUDIO));
}

void* SdlAudioRenderer::getAudioBuffer(int*)
{
    return m_AudioBuffer;
}

bool SdlAudioRenderer::submitAudio(int bytesWritten)
{
    if (bytesWritten == 0) {
        // Nothing to do
        return true;
    }

    // Bursty packet delivery is common on Wi-Fi. Only drop to catch up when
    // both the pre-decode queue is beyond the configured tolerance and the
    // output queue has enough audio to avoid turning that drop into an underrun.
    int queueSize = SDL_GetQueuedAudioSize(m_AudioDevice);
    int queuedAudioMs = queueSize / m_FrameSize * m_FrameDurationMs;
    if (LiGetPendingAudioDuration() + queuedAudioMs >= m_JitterBufferMs && queuedAudioMs >= 15) {
        ++m_DropCount;
        m_HadProducerDrop = true;
        return true;
    }

    // Provide backpressure on the queue to ensure too many frames don't build up
    // in SDL's audio queue, but don't wait forever to avoid a deadlock if the
    // audio device fails.
    bool canQueue = false;
    for (int i = 0; i < 100; i++) {
        // Our device may enter a permanent error status upon removal, so we need
        // to recreate the audio device to pick up the new default audio device.
        if (SDL_GetAudioDeviceStatus(m_AudioDevice) == SDL_AUDIO_STOPPED) {
            return false;
        }

        // Only queue more samples where there is 50 ms or less in SDL's queue
        queueSize = SDL_GetQueuedAudioSize(m_AudioDevice);
        if (queueSize / m_FrameSize * m_FrameDurationMs <= 50) {
            m_QueuedAudioSize.store(queueSize + bytesWritten);
            canQueue = true;
            break;
        }

        SDL_Delay(1);
    }

    if (!canQueue) {
        ++m_DropCount;
        m_HadProducerDrop = true;
        return true;
    }

    const uint32_t framesWritten = bytesWritten /
        (m_ChannelCount * getAudioBufferSampleSize());
    float *samples = (float *)m_AudioBuffer;
    if (m_HadProducerDrop && framesWritten != 0) {
        crossfadeFromLast(samples,
                          framesWritten,
                          m_ChannelCount,
                          m_FadeFrames,
                          m_LastQueuedSamples);
        m_HadProducerDrop = false;
    }

    if (SDL_QueueAudio(m_AudioDevice, m_AudioBuffer, bytesWritten) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "Failed to queue audio sample: %s",
                     SDL_GetError());
        return false;
    }

    if (framesWritten != 0) {
        const float *lastFrame = samples + (framesWritten - 1) * m_ChannelCount;
        for (uint32_t channel = 0; channel < m_ChannelCount; channel++) {
            m_LastQueuedSamples[channel] = lastFrame[channel];
        }
    }

    return true;
}

IAudioRenderer::AudioFormat SdlAudioRenderer::getAudioBufferFormat()
{
    return AudioFormat::Float32NE;
}

void SdlAudioRenderer::updateMetrics()
{
    DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
        metrics.audioDropCount += m_DropCount.exchange(0);
        metrics.audioInBufferMs = (float)m_QueuedAudioSize.load() / m_FrameSize * m_FrameDurationMs;
    });
}

void SdlAudioRenderer::notifyAudioDiscontinuity()
{
    m_HadProducerDrop = true;
}
