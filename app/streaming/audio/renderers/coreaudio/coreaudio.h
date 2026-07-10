#pragma once

#include "../renderer.h"
#include "au_spatial_renderer.h"
#include "AllocatedAudioBufferList.h"
#include "TPCircularBuffer.h"

#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioToolbox.h>

#include <atomic>
#include <vector>

class CoreAudioRenderer : public IAudioRenderer
{
public:
    CoreAudioRenderer();
    ~CoreAudioRenderer();

    bool prepareForPlayback(const OPUS_MULTISTREAM_CONFIGURATION* opusConfig) override;
    virtual void updateMetrics() override;
    virtual void* getAudioBuffer(int* size) override;
    virtual bool submitAudio(int bytesWritten) override;
    virtual void notifyAudioDiscontinuity() override;
    virtual int getCapabilities();
    virtual AudioFormat getAudioBufferFormat() override;
    virtual void setHeadTracking(bool enabled) override;
    const char * getRendererName() { return "CoreAudio"; }

    friend OSStatus renderCallbackDirect(void *, AudioUnitRenderActionFlags *, const AudioTimeStamp *, uint32_t, uint32_t, AudioBufferList *);
    friend OSStatus renderCallbackSpatial(void *, AudioUnitRenderActionFlags *, const AudioTimeStamp *, uint32_t, uint32_t, AudioBufferList *);
    friend OSStatus onDeviceOverload(AudioObjectID, UInt32, const AudioObjectPropertyAddress *, void *);
    friend OSStatus onAudioNeedsReinit(AudioObjectID, UInt32, const AudioObjectPropertyAddress *, void *);
    friend OSStatus onAudioNeedsReinit(UInt32, AudioObjectID, UInt32, const AudioObjectPropertyAddress *, void *);;

private:
    bool initAudioUnit();
    bool initRingBuffer();
    bool initListeners();
    void deinitListeners();
    bool setCallback(AURenderCallback);
    void clearCallback();
    void cleanup();
    AUSpatialMixerOutputType getSpatialMixerOutputType();
    void setOutputDeviceName(CFStringRef);

    AudioUnit m_OutputAU = nullptr;
    AUSpatialRenderer m_SpatialAU;

    // output device metadata
    AudioDeviceID m_OutputDeviceID = 0;
    AudioStreamBasicDescription m_OutputASBD{};
    char *m_OutputDeviceName = nullptr;
    char m_OutputTransportType[5]{};
    char m_OutputDataSource[5]{};
    const OPUS_MULTISTREAM_CONFIGURATION* m_opusConfig = nullptr;

    // buffers
    TPCircularBuffer m_RingBuffer{};
    AllocatedAudioBufferList m_SpatialBuffer;
    std::vector<float> m_DecodeBuffer;
    std::vector<float> m_LastOutputSamples;
    std::vector<float> m_LastQueuedSamples;
    double m_AudioPacketDuration = 0.0;
    uint32_t m_BufferFrameSize = 0;
    uint32_t m_BytesPerFrame = 0;
    uint32_t m_RebufferThresholdBytes = 0;
    uint32_t m_FadeFrames = 0;
    int m_JitterBufferMs = 80;

    // latency
    double m_OutputHardwareLatency = 0.0;
    double m_TotalSoftwareLatency = 0.0;
    double m_OutputSoftwareLatencyMin = 0.0;
    double m_OutputSoftwareLatencyMax = 0.0;

    // internal device state
    std::atomic<bool> m_needsReinit{false};
    bool m_OutputInitialized = false;
    bool m_ListenersInitialized = false;
    bool m_Spatial = false;
    bool m_Rebuffering = true;
    bool m_FadeInPending = true;
    bool m_HasLastOutput = false;
    bool m_HadProducerDrop = false;
    uint32_t m_SpatialOutputType = 0;

    // stats
    double m_LastSampleTime = 0.0;
    uint32_t m_LastNumFrames = 0;
    uint32_t m_BufferSize = 0;
    bool m_SeenAudio = true;
    void statsIncDeviceOverload();
    void statsTrackRender(uint64_t, const AudioTimeStamp *, uint32_t, bool);
    std::atomic<uint32_t> m_DropCount{0};
    std::atomic<uint32_t> m_DropCountUnderrun{0};
    std::atomic<int> m_QueuedAudioSize;
};
