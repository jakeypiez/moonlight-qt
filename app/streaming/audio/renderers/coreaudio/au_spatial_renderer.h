#pragma once

#include "TPCircularBuffer.h"

#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioToolbox.h>

#include <Limelight.h>

#include <atomic>
#include <vector>

typedef void (^SimpleBlock)();

class AUSpatialRenderer
{
public:
    AUSpatialRenderer();
    ~AUSpatialRenderer();
    void cleanup();
    void clearCallback();

    double getAudioUnitLatency();
    bool getHeadTracking();
    void setHeadTracking(bool enabled);
    void setRingBufferPtr(TPCircularBuffer* __nonnull buffer);
    void setStatsTrackRenderBlock(SimpleBlock _Nonnull);
    bool setup(AUSpatialMixerOutputType outputType, float sampleRate, int inChannelCount, int samplesPerFrame);
    OSStatus setStreamFormatAndACL(float inSampleRate, AudioChannelLayoutTag inLayoutTag, AudioUnitScope inScope, AudioUnitElement inElement);
    OSStatus setOutputType(AUSpatialMixerOutputType outputType);
    OSStatus process(AudioBufferList* __nullable outputABL, AudioUnitRenderActionFlags* __nonnull ioActionFlags, const AudioTimeStamp* __nullable inTimestamp, uint32_t inNumberFrames);
    bool takeUnderrun();

    friend OSStatus inputCallback(void * _Nonnull,
                    AudioUnitRenderActionFlags *_Nullable,
                    const AudioTimeStamp * _Nullable,
                    uint32_t, uint32_t,
                    AudioBufferList * _Nonnull);

    uint32_t m_PersonalizedHRTF;

private:
    AudioUnit _Nullable m_Mixer = nullptr;
    TPCircularBuffer* _Nullable m_RingBufferPtr = nullptr; // pointer to RingBuffer in the outer CoreAudioRenderer
    SimpleBlock _Nullable m_StatsTrackRenderBlock = nullptr;

    bool m_Initialized = false;
    bool m_Rebuffering = true;
    bool m_FadeInPending = true;
    bool m_HasLastInput = false;
    int m_InputChannelCount = 0;
    uint32_t m_RebufferThresholdBytes = 0;
    uint32_t m_FadeFrames = 0;
    std::vector<float> m_LastInputSamples;
    std::atomic<bool> m_DidUnderrun{false};
    double m_AudioUnitLatency = 0.0;
};
