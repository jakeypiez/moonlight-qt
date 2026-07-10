#include "coreaudio.h"
#include "coreaudio_helpers.h"
#include "imgui/devui.h"
#include "settings/streamingpreferences.h"

#if TARGET_OS_OSX
#include <IOKit/audio/IOAudioTypes.h>
#endif

#include <QtGlobal>
#include <SDL.h>
#include <cmath>
#include <string>

namespace {

constexpr double kMinimumRingBufferSeconds = 0.060;
constexpr int kMaximumLocalQueueMs = 50;
constexpr double kMinimumRebufferSeconds = 0.010;
constexpr double kMaximumRebufferSeconds = 0.020;
constexpr double kTransitionFadeSeconds = 0.001;
constexpr uint32_t kSpatialBufferFrames = 4096;

void fadeInterleavedIn(float* samples, uint32_t frames, uint32_t channels, uint32_t fadeFrames)
{
    fadeFrames = qMin(frames, fadeFrames);
    if (fadeFrames == 0) {
        return;
    }

    for (uint32_t frame = 0; frame < fadeFrames; frame++) {
        const float gain = fadeFrames == 1 ? 1.0f : (float)frame / (fadeFrames - 1);
        for (uint32_t channel = 0; channel < channels; channel++) {
            samples[frame * channels + channel] *= gain;
        }
    }
}

void fadeInterleavedOut(float* samples, uint32_t frames, uint32_t channels, uint32_t fadeFrames)
{
    fadeFrames = qMin(frames, fadeFrames);
    if (fadeFrames == 0) {
        return;
    }

    const uint32_t firstFadeFrame = frames - fadeFrames;
    for (uint32_t frame = 0; frame < fadeFrames; frame++) {
        const float gain = fadeFrames == 1 ? 0.0f : 1.0f - (float)frame / (fadeFrames - 1);
        for (uint32_t channel = 0; channel < channels; channel++) {
            samples[(firstFadeFrame + frame) * channels + channel] *= gain;
        }
    }
}

void fadeInterleavedFromLast(float* samples,
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
        const float gain = fadeFrames == 1 ? 0.0f : 1.0f - (float)frame / (fadeFrames - 1);
        for (uint32_t channel = 0; channel < channels; channel++) {
            samples[frame * channels + channel] = lastSamples[channel] * gain;
        }
    }
}

void crossfadeInterleavedFromLast(float* samples,
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

CoreAudioRenderer::CoreAudioRenderer()
    : m_SpatialBuffer(2, 4096),
    m_QueuedAudioSize{0}
{
    DEBUG_TRACE("CoreAudioRenderer construct");

    AudioComponentDescription description;
    description.componentType = kAudioUnitType_Output;
#if TARGET_OS_IPHONE
    description.componentSubType = kAudioUnitSubType_RemoteIO;
#elif TARGET_OS_OSX
    description.componentSubType = kAudioUnitSubType_HALOutput;
#endif
    description.componentManufacturer = kAudioUnitManufacturer_Apple;
    description.componentFlags = 0;
    description.componentFlagsMask = 0;

    AudioComponent comp = AudioComponentFindNext(NULL, &description);
    if (!comp) {
        return;
    }

    OSStatus status = AudioComponentInstanceNew(comp, &m_OutputAU);
    if (status != noErr) {
        CA_LogError(status, "Failed to create an instance of HALOutput or RemoteIO");
        throw std::runtime_error("Failed to create an instance of HALOutput or RemoteIO");
    }
}

CoreAudioRenderer::~CoreAudioRenderer()
{
    DEBUG_TRACE("CoreAudioRenderer destruct");
    cleanup();
}

void CoreAudioRenderer::cleanup()
{
    DEBUG_TRACE("CoreAudioRenderer cleanup");

    if (m_ListenersInitialized) {
        deinitListeners();
        m_ListenersInitialized = false;
    }
    m_OutputDeviceID = 0;

    if (m_OutputAU != nullptr) {
        AudioOutputUnitStop(m_OutputAU);
        if (m_OutputInitialized) {
            AudioUnitUninitialize(m_OutputAU);
            m_OutputInitialized = false;
        }
        clearCallback();

        m_SpatialAU.cleanup();

        AudioComponentInstanceDispose(m_OutputAU);
        m_OutputAU = nullptr;
    }

    if (m_RingBuffer.buffer != nullptr) {
        TPCircularBufferCleanup(&m_RingBuffer);
    }

    if (m_OutputDeviceName) {
        free(m_OutputDeviceName);
        m_OutputDeviceName = nullptr;
    }
}

int CoreAudioRenderer::getCapabilities()
{
    // CAPABILITY_DIRECT_SUBMIT feels worse than decoding in a separate thread
    return CAPABILITY_SUPPORTS_ARBITRARY_AUDIO_DURATION;
}

IAudioRenderer::AudioFormat CoreAudioRenderer::getAudioBufferFormat()
{
    return AudioFormat::Float32NE;
}

void CoreAudioRenderer::statsIncDeviceOverload()
{
   //m_ActiveWndAudioStats.totalGlitches++;
}

// realtime method
void CoreAudioRenderer::statsTrackRender(uint64_t startTimeUs, const AudioTimeStamp *inTimestamp, uint32_t inNumberFrames, bool didUnderrun)
{
    Q_UNUSED(startTimeUs);

    // If no audio is playing, it's normal to get many underruns in a row.
    // We will only increment the underrun stat for the first empty buffer and
    // require some audio data before counting it again.
    if (didUnderrun) {
        if (m_SeenAudio) {
            ++m_DropCountUnderrun;
            m_SeenAudio = false;
        }
    } else {
        m_SeenAudio = true;
    }

    // check for lost packets because we weren't called in time, possibly due to system overload
    if (m_LastSampleTime && inTimestamp->mFlags & kAudioTimeStampSampleTimeValid) {
        double expectedSampleTime = m_LastSampleTime + m_LastNumFrames;
        if (expectedSampleTime != inTimestamp->mSampleTime) {
            //m_ActiveWndAudioStats.totalGlitches++;

// #ifdef COREAUDIO_DEBUG
//             dispatch_async(dispatch_get_main_queue(), ^{
//                 DEBUG_TRACE("[%llu] Error: lost/dropped audio frames: %u (%.02fms)", inTimestamp->mHostTime, lostFrames, lostDuration * 1000.0);
//             });
// #endif
        }
    }

    m_LastSampleTime = inTimestamp->mSampleTime;
    m_LastNumFrames = inNumberFrames;

    // add this to our decoderTime
    //uint64_t decodeTimeUs = LiGetMicroseconds() - startTimeUs;
    //m_ActiveWndAudioStats.decodeDurationUs += decodeTimeUs;

    // We now have decodeDurationUs covering 2 time periods:
    // 1. Filling the queue: Opus decoding plus write to circular buffer (statsTrackDecodeTime)
    // 2. Emptying the queue: from start of AudioUnit callback in either direct or spatial mode (statsTrackRender)
    //    Although it's referred to as render time, the time is just added to decodeDurationUs
}

// realtime method
OSStatus renderCallbackDirect(void *inRefCon,
                                     AudioUnitRenderActionFlags *ioActionFlags,
                                     const AudioTimeStamp *inTimestamp,
                                     uint32_t /*inBusNumber*/,
                                     uint32_t inNumberFrames,
                                     AudioBufferList *ioData)
{
    uint64_t start = LiGetMicroseconds();

    CoreAudioRenderer *me = (CoreAudioRenderer *)inRefCon;
    bool didUnderrun = false;

    if (ioActionFlags == nullptr) {
        return kAudio_ParamError;
    }

    if (ioData == nullptr || ioData->mNumberBuffers != 1 ||
            ioData->mBuffers[0].mData == nullptr || me->m_BytesPerFrame == 0) {
        if (ioData != nullptr) {
            for (uint32_t i = 0; i < ioData->mNumberBuffers; i++) {
                if (ioData->mBuffers[i].mData != nullptr) {
                    memset(ioData->mBuffers[i].mData, 0, ioData->mBuffers[i].mDataByteSize);
                }
            }
        }
        *ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        me->m_Rebuffering = true;
        me->m_FadeInPending = true;
        me->statsTrackRender(start, inTimestamp, inNumberFrames, true);
        return noErr;
    }

    AudioBuffer& output = ioData->mBuffers[0];
    float *targetBuffer = (float *)output.mData;
    const uint32_t channelCount = me->m_opusConfig->channelCount;
    const uint32_t requestedBytes = qMin(output.mDataByteSize,
                                         inNumberFrames * me->m_BytesPerFrame) /
                                    me->m_BytesPerFrame * me->m_BytesPerFrame;
    const uint32_t requestedFrames = requestedBytes / me->m_BytesPerFrame;
    memset(targetBuffer, 0, output.mDataByteSize);

    // Pull audio from playthrough buffer
    uint32_t availableBytes = 0;
    float *buffer = (float *)TPCircularBufferTail(&me->m_RingBuffer, &availableBytes);

    // Start and resume only after enough PCM is queued to absorb normal packet
    // arrival jitter. Once running, don't stop until we actually underrun.
    if (me->m_Rebuffering &&
            (availableBytes < me->m_RebufferThresholdBytes || availableBytes < requestedBytes)) {
        *ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        me->m_QueuedAudioSize.store(availableBytes);
        me->statsTrackRender(start, inTimestamp, inNumberFrames, true);
        return noErr;
    }

    if (me->m_Rebuffering) {
        me->m_Rebuffering = false;
        me->m_FadeInPending = true;
    }

    const uint32_t availableFrames = availableBytes / me->m_BytesPerFrame;
    const uint32_t framesCopied = qMin(requestedFrames, availableFrames);
    const uint32_t bytesCopied = framesCopied * me->m_BytesPerFrame;

    if (bytesCopied != 0) {
        memcpy(targetBuffer, buffer, bytesCopied);
        TPCircularBufferConsume(&me->m_RingBuffer, bytesCopied);

        if (me->m_FadeInPending) {
            fadeInterleavedIn(targetBuffer, framesCopied, channelCount, me->m_FadeFrames);
            me->m_FadeInPending = false;
        }
    }

    if (framesCopied < requestedFrames) {
        didUnderrun = true;
        me->m_Rebuffering = true;
        me->m_FadeInPending = true;

        if (framesCopied != 0) {
            fadeInterleavedOut(targetBuffer, framesCopied, channelCount, me->m_FadeFrames);
        }
        else if (me->m_HasLastOutput) {
            fadeInterleavedFromLast(targetBuffer,
                                    requestedFrames,
                                    channelCount,
                                    me->m_FadeFrames,
                                    me->m_LastOutputSamples);
        }
        else {
            *ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        }

        me->m_HasLastOutput = false;
    }
    else if (framesCopied != 0) {
        const float *lastFrame = targetBuffer + (framesCopied - 1) * channelCount;
        for (uint32_t channel = 0; channel < channelCount; channel++) {
            me->m_LastOutputSamples[channel] = lastFrame[channel];
        }
        me->m_HasLastOutput = true;
    }

    me->m_QueuedAudioSize.store(availableBytes - bytesCopied);
    me->statsTrackRender(start, inTimestamp, inNumberFrames, didUnderrun);

    return noErr;
}

// realtime method
OSStatus renderCallbackSpatial(void *inRefCon,
                                      AudioUnitRenderActionFlags *ioActionFlags,
                                      const AudioTimeStamp *inTimestamp,
                                      uint32_t /*inBusNumber*/,
                                      uint32_t inNumberFrames,
                                      AudioBufferList *ioData)
{
    uint64_t start = LiGetMicroseconds();
    CoreAudioRenderer *me = (CoreAudioRenderer *)inRefCon;
    AudioBufferList *spatialBuffer = me->m_SpatialBuffer.get();

    if (ioData == nullptr || ioActionFlags == nullptr) {
        return kAudio_ParamError;
    }

    if (inNumberFrames > kSpatialBufferFrames) {
        for (uint32_t i = 0; i < ioData->mNumberBuffers; i++) {
            if (ioData->mBuffers[i].mData != nullptr) {
                memset(ioData->mBuffers[i].mData, 0, ioData->mBuffers[i].mDataByteSize);
            }
        }
        *ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        me->statsTrackRender(start, inTimestamp, inNumberFrames, true);
        return noErr;
    }

    // Set the byte size with the output audio buffer list.
    for (uint32_t i = 0; i < spatialBuffer->mNumberBuffers; i++) {
        spatialBuffer->mBuffers[i].mDataByteSize = inNumberFrames * sizeof(float);
        memset(spatialBuffer->mBuffers[i].mData, 0, spatialBuffer->mBuffers[i].mDataByteSize);
    }

    // Process the input frames with the audio unit spatial mixer.
    OSStatus status = me->m_SpatialAU.process(spatialBuffer, ioActionFlags, inTimestamp, inNumberFrames);
    bool didUnderrun = me->m_SpatialAU.takeUnderrun();

    // Copy the temporary buffer to the output.
    for (uint32_t i = 0; i < ioData->mNumberBuffers; i++) {
        AudioBuffer& output = ioData->mBuffers[i];
        if (output.mData == nullptr) {
            didUnderrun = true;
            continue;
        }
        const uint32_t bytesToCopy = qMin(output.mDataByteSize, inNumberFrames * (uint32_t)sizeof(float));
        if (status == noErr && i < spatialBuffer->mNumberBuffers) {
            memcpy(output.mData, spatialBuffer->mBuffers[i].mData, bytesToCopy);
            if (bytesToCopy < output.mDataByteSize) {
                memset((char *)output.mData + bytesToCopy, 0, output.mDataByteSize - bytesToCopy);
            }
        }
        else {
            memset(output.mData, 0, output.mDataByteSize);
        }
    }

    if (status != noErr) {
        *ioActionFlags |= kAudioUnitRenderAction_OutputIsSilence;
        didUnderrun = true;
    }

    uint32_t availableBytes = 0;
    TPCircularBufferTail(&me->m_RingBuffer, &availableBytes);
    me->m_QueuedAudioSize.store(availableBytes);

    me->statsTrackRender(start, inTimestamp, inNumberFrames, didUnderrun);

    return noErr;
}

bool CoreAudioRenderer::prepareForPlayback(const OPUS_MULTISTREAM_CONFIGURATION* opusConfig)
{
    OSStatus status = noErr;
    m_opusConfig = opusConfig;

    if (m_OutputAU == nullptr || opusConfig == nullptr || opusConfig->sampleRate <= 0 ||
            opusConfig->samplesPerFrame <= 0 || opusConfig->channelCount <= 0) {
        return false;
    }

    StreamingPreferences *prefs = StreamingPreferences::get();
    m_JitterBufferMs = qBound(30, prefs->audioJitterBufferMs, 150);

    // Request the OS set our buffer close to the Opus packet size
    m_AudioPacketDuration = (double)opusConfig->samplesPerFrame / opusConfig->sampleRate;
    m_BytesPerFrame = opusConfig->channelCount * sizeof(float);

    const double rebufferSeconds = qBound(kMinimumRebufferSeconds,
                                           m_AudioPacketDuration * 3.0,
                                           kMaximumRebufferSeconds);
    const uint32_t rebufferFrames = (uint32_t)ceil(rebufferSeconds * opusConfig->sampleRate);
    m_RebufferThresholdBytes = rebufferFrames * m_BytesPerFrame;
    m_FadeFrames = qMax(1U, (uint32_t)ceil(kTransitionFadeSeconds * opusConfig->sampleRate));
    m_Rebuffering = true;
    m_FadeInPending = true;
    m_HasLastOutput = false;
    m_HadProducerDrop = false;

    const size_t samplesPerPacket = (size_t)opusConfig->samplesPerFrame * opusConfig->channelCount;
    m_DecodeBuffer.resize(samplesPerPacket);
    m_LastOutputSamples.assign(opusConfig->channelCount, 0.0f);
    m_LastQueuedSamples.assign(opusConfig->channelCount, 0.0f);

    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                "CoreAudioRenderer jitter buffer: %d ms; rebuffer target: %.1f ms",
                m_JitterBufferMs,
                rebufferSeconds * 1000.0);

    if (!initAudioUnit()) {
        DEBUG_TRACE("initAudioUnit failed");
        return false;
    }

    if (!initRingBuffer()) {
        DEBUG_TRACE("initRingBuffer failed");
        return false;
    }

    // Cleanup removes all listener registrations and safely tolerates selectors
    // that were not added, so mark this before the first incremental add.
    m_ListenersInitialized = true;
    if (!initListeners()) {
        DEBUG_TRACE("initListeners failed");
        return false;
    }

    m_Spatial = false;
    AUSpatialMixerOutputType outputType = getSpatialMixerOutputType();

    DEBUG_TRACE("CoreAudioRenderer getSpatialMixerOutputType = %d", outputType);

    if (opusConfig->channelCount > 2) {
        if (outputType != kSpatialMixerOutputType_ExternalSpeakers) {
            m_Spatial = true;
        }
    }

    if (prefs->spatialAudioConfig == StreamingPreferences::SAC_DISABLED) {
        // User has disabled spatial audio
        DEBUG_TRACE("CoreAudioRenderer user has disabled spatial audio");
        m_Spatial = false;
    }

    // indicate the format our callback will provide samples in
    // If necessary, the OS takes care of resampling (but not downmixing, hmm)
    AudioStreamBasicDescription streamDesc;
    memset(&streamDesc, 0, sizeof(AudioStreamBasicDescription));
    streamDesc.mSampleRate       = opusConfig->sampleRate;
    streamDesc.mFormatID         = kAudioFormatLinearPCM;
    streamDesc.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    streamDesc.mFramesPerPacket  = 1;
    streamDesc.mChannelsPerFrame = (uint32_t)opusConfig->channelCount;
    streamDesc.mBitsPerChannel   = 32;
    streamDesc.mBytesPerPacket   = 4 * opusConfig->channelCount;
    streamDesc.mBytesPerFrame    = streamDesc.mBytesPerPacket;

    if (m_Spatial) {
        // render audio for binaural headphones or built-in laptop speakers
        if (!setCallback(renderCallbackSpatial)) {
            return false;
        }

        // The spatial mixer consumes the multichannel ring buffer internally,
        // but its output to the HAL is always non-interleaved stereo.
        streamDesc.mFormatFlags    |= kAudioFormatFlagIsNonInterleaved;
        streamDesc.mChannelsPerFrame = 2;
        streamDesc.mBytesPerPacket = 4;
        streamDesc.mBytesPerFrame  = 4;

        m_SpatialOutputType = outputType;

        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "CoreAudioRenderer is using spatial audio output");

        if (!m_SpatialAU.setup(outputType,
                               opusConfig->sampleRate,
                               opusConfig->channelCount,
                               opusConfig->samplesPerFrame)) {
            DEBUG_TRACE("m_SpatialAU.setup failed");
            return false;
        }

        m_TotalSoftwareLatency += m_SpatialAU.getAudioUnitLatency();
    } else {
        // direct passthrough of all channels for stereo and HDMI
        if (!setCallback(renderCallbackDirect)) {
            return false;
        }

        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "CoreAudioRenderer is using passthrough mode");
    }

    status = AudioUnitSetProperty(m_OutputAU, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &streamDesc, sizeof(streamDesc));
    if (status != noErr) {
        CA_LogError(status, "Failed to set output stream format");
        return false;
    }

    uint32_t maximumFramesPerSlice = kSpatialBufferFrames;
    status = AudioUnitSetProperty(m_OutputAU,
                                  kAudioUnitProperty_MaximumFramesPerSlice,
                                  kAudioUnitScope_Global,
                                  0,
                                  &maximumFramesPerSlice,
                                  sizeof(maximumFramesPerSlice));
    if (status != noErr) {
        // Some output devices expose this as read-only. The callback still
        // guards the fixed spatial scratch capacity, so retain compatibility.
        CA_LogError(status, "Unable to set maximum output frames per slice; using device default");
    }

    // Configure the device, stream format, and callback before initializing.
    status = AudioUnitInitialize(m_OutputAU);
    if (status != noErr) {
        CA_LogError(status, "Failed to initialize the output audio unit");
        return false;
    }
    m_OutputInitialized = true;

    DEBUG_TRACE("CoreAudioRenderer start");
    status = AudioOutputUnitStart(m_OutputAU);
    if (status != noErr) {
        CA_LogError(status, "Failed to start output audio unit");
        return false;
    }

    DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
        metrics.opusChannelCount = m_opusConfig->channelCount;
        strncpy(metrics.audioOutputDeviceName, m_OutputDeviceName, sizeof(metrics.audioOutputDeviceName));
        metrics.audioFrameDurationMs = m_AudioPacketDuration * 1000;
        metrics.audioSampleRate = m_OutputASBD.mSampleRate;
        metrics.audioChannels = m_OutputASBD.mChannelsPerFrame;
        metrics.spatialAudioActive = m_Spatial;
        metrics.audioPersonalizedHRTF = m_SpatialAU.m_PersonalizedHRTF;
        metrics.audioHeadTracking = m_SpatialAU.getHeadTracking();
        strncpy(metrics.audioOutputTransportType, m_OutputTransportType, 5);
        strncpy(metrics.audioOutputDataSource, m_OutputDataSource, 5);
        metrics.audioTotalSoftwareLatency = m_TotalSoftwareLatency;
        metrics.audioOutputHardwareLatency = m_OutputHardwareLatency;
    });

    return true;
}

void CoreAudioRenderer::updateMetrics()
{
    DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
        metrics.audioDropCount += m_DropCount.exchange(0);
        metrics.audioDropCountUnderrun += m_DropCountUnderrun.exchange(0);
        int bytesPerMs = (m_opusConfig->sampleRate * m_opusConfig->channelCount * sizeof(float)) / 1000;
        metrics.audioInBufferMs = (float)m_QueuedAudioSize.load() / bytesPerMs;
    });
}

bool CoreAudioRenderer::initAudioUnit()
{
    OSStatus status = noErr;

    /* macOS:
     * disable OutputAU input IO
     * enable OutputAU output IO
     * get system default output AudioDeviceID  (todo: allow user to choose specific device from list)
     * set OutputAU to AudioDeviceID
     * get device's AudioStreamBasicDescription (format, bit depth, samplerate, etc)
     * get device name
     * get output buffer frame size
     * get output buffer min/max
     * set output buffer frame size
     */

#if TARGET_OS_OSX
    constexpr AudioUnitElement outputElement{0};
    constexpr AudioUnitElement inputElement{1};

    {
        uint32_t enableIO = 0;
        status = AudioUnitSetProperty(m_OutputAU, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, inputElement, &enableIO, sizeof(enableIO));
        if (status != noErr) {
            CA_LogError(status, "Failed to disable the input on AUHAL");
            return false;
        }

        enableIO = 1;
        status = AudioUnitSetProperty(m_OutputAU, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, outputElement, &enableIO, sizeof(enableIO));
        if (status != noErr) {
            CA_LogError(status, "Failed to enable the output on AUHAL");
            return false;
        }
    }

    {
        uint32_t size = sizeof(AudioDeviceID);
        AudioObjectPropertyAddress addr{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, outputElement, nil, &size, &m_OutputDeviceID);
        if (status != noErr) {
            CA_LogError(status, "Failed to get the default output device");
            return false;
        }
    }

    {
        CFStringRef name;
        uint32_t nameSize = sizeof(CFStringRef);
        AudioObjectPropertyAddress addr{kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addr, 0, nil, &nameSize, &name);
        if (status != noErr) {
            CA_LogError(status, "Failed to get name of output device");
            return false;
        }
        setOutputDeviceName(name);
        CFRelease(name);
        DEBUG_TRACE("CoreAudioRenderer default output device ID: %d, name: %s", m_OutputDeviceID, m_OutputDeviceName);
    }

    {
        // Set the current device to the default output device.
        // This should be done only after I/O is enabled on the output audio unit.
        status = AudioUnitSetProperty(m_OutputAU, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, outputElement, &m_OutputDeviceID, sizeof(AudioDeviceID));
        if (status != noErr) {
            CA_LogError(status, "Failed to set the default output device");
            return false;
        }
    }

    {
        uint32_t streamFormatSize = sizeof(AudioStreamBasicDescription);
        AudioObjectPropertyAddress addr{kAudioDevicePropertyStreamFormat, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addr, 0, nil, &streamFormatSize, &m_OutputASBD);
        if (status != noErr) {
            CA_LogError(status, "Failed to get output device AudioStreamBasicDescription");
            return false;
        }
        CA_PrintASBD("CoreAudioRenderer output format:", &m_OutputASBD);
    }

    // Buffer:
    // The goal here is to set the system buffer to our desired value, which is currently in m_AudioPacketDuration.
    // First we get the current value, and the range of allowed values, set our value, and then query to find the actual value.
    // We also query the hardware latency (e.g. Bluetooth delay for AirPods), but this is just for fun

    {
        uint32_t bufferFrameSize = 0;
        uint32_t size = sizeof(uint32_t);
        AudioObjectPropertyAddress addr{kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addr, 0, nil, &size, &bufferFrameSize);
        if (status != noErr) {
            CA_LogError(status, "Failed to get the output device buffer frame size");
            return false;
        }
        DEBUG_TRACE("CoreAudioRenderer output current BufferFrameSize %d", bufferFrameSize);
    }

    {
        AudioValueRange avr;
        uint32_t size = sizeof(AudioValueRange);
        AudioObjectPropertyAddress addr{kAudioDevicePropertyBufferFrameSizeRange, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addr, 0, nil, &size, &avr);
        if (status != noErr) {
            CA_LogError(status, "Failed to get the output device buffer frame size range");
            return false;
        }
        m_OutputSoftwareLatencyMin = avr.mMinimum / m_OutputASBD.mSampleRate;
        m_OutputSoftwareLatencyMax = avr.mMaximum / m_OutputASBD.mSampleRate;
        DEBUG_TRACE("CoreAudioRenderer output BufferFrameSizeRange: %.0f - %.0f", avr.mMinimum, avr.mMaximum);
    }

    // The latency values we have access to are:
    // kAudioDevicePropertyBufferFrameSize    our requested buffer as close to Opus packet size as possible
    //   + kAudioDevicePropertySafetyOffset   an additional CoreAudio buffer
    //   + kAudioUnitProperty_Latency         processing latency of OutputAU (+ SpatialAU in spatial mode)
    //   = total software latency
    // kAudioDevicePropertyLatency = hardware latency

    {
        double desiredBufferFrameSize = m_AudioPacketDuration;
        desiredBufferFrameSize = qMax(qMin(desiredBufferFrameSize, m_OutputSoftwareLatencyMax), m_OutputSoftwareLatencyMin);
        uint32_t bufferFrameSize = (uint32_t)(desiredBufferFrameSize * m_OutputASBD.mSampleRate);
        AudioObjectPropertyAddress addrSet{kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectSetPropertyData(m_OutputDeviceID, &addrSet, 0, NULL, sizeof(uint32_t), &bufferFrameSize);
        if (status != noErr) {
            // Bluetooth, aggregate, and virtual devices may not allow clients
            // to change this value. Continue with the current device size.
            CA_LogError(status, "Unable to set output device buffer frame size; using device default");
        }
        else {
            DEBUG_TRACE("CoreAudioRenderer output requested BufferFrameSize of %d (%0.3f ms)", bufferFrameSize, desiredBufferFrameSize * 1000.0);
        }

        // see what we got
        uint32_t size = sizeof(uint32_t);
        AudioObjectPropertyAddress addrGet{kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addrGet, 0, nil, &size, &m_BufferFrameSize);
        if (status != noErr) {
            CA_LogError(status, "Failed to get the output device buffer frame size");
            return false;
        }
        double bufferFrameLatency = (double)m_BufferFrameSize / m_OutputASBD.mSampleRate;
        m_TotalSoftwareLatency += bufferFrameLatency;
        m_TotalSoftwareLatency += 0.0025; // Opus has 2.5ms of initial delay
        DEBUG_TRACE("CoreAudioRenderer output now has actual BufferFrameSize of %d (%0.3f ms)", m_BufferFrameSize, bufferFrameLatency * 1000.0);
    }

    {
        double audioUnitLatency = 0.0;
        uint32_t size = sizeof(audioUnitLatency);
        status = AudioUnitGetProperty(m_OutputAU, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0, &audioUnitLatency, &size);
        if (status != noErr) {
            CA_LogError(status, "Failed to get OutputAU AudioUnit latency");
            return false;
        }
        m_TotalSoftwareLatency += audioUnitLatency;
        DEBUG_TRACE("CoreAudioRenderer OutputAU AudioUnit latency: %0.2f ms", audioUnitLatency * 1000.0);
    }

    {
        uint32_t safetyOffsetLatency = 0;
        uint32_t size = sizeof(safetyOffsetLatency);
        AudioObjectPropertyAddress addrGet{kAudioDevicePropertySafetyOffset, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addrGet, 0, nil, &size, &safetyOffsetLatency);
        if (status != noErr) {
            CA_LogError(status, "Failed to get safety offset latency");
            return false;
        }
        m_TotalSoftwareLatency += (double)safetyOffsetLatency / m_OutputASBD.mSampleRate;
        DEBUG_TRACE("CoreAudioRenderer OutputAU safety latency: %0.2f ms", ((double)safetyOffsetLatency / m_OutputASBD.mSampleRate) * 1000.0);
    }

    {
        uint32_t latencyFrames;
        uint32_t size = sizeof(uint32_t);
        AudioObjectPropertyAddress addr{kAudioDevicePropertyLatency, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
        status = AudioObjectGetPropertyData(m_OutputDeviceID, &addr, 0, nil, &size, &latencyFrames);
        if (status != noErr) {
            CA_LogError(status, "Failed to get the output device hardware latency");
            return false;
        }
        m_OutputHardwareLatency = (double)latencyFrames / m_OutputASBD.mSampleRate;
        DEBUG_TRACE("CoreAudioRenderer output hardware latency: %d (%0.2f ms)", latencyFrames, m_OutputHardwareLatency * 1000.0);
    }
#endif

    return true;
}

bool CoreAudioRenderer::initRingBuffer()
{
    // Keep decoded PCM bounded independently of the configurable pre-decode
    // jitter window. submitAudio() applies backpressure above 50 ms so network
    // bursts remain in Moonlight's packet queue instead of becoming stale PCM.
    int packetsToBuffer = qMax(4, (int)ceil(kMinimumRingBufferSeconds / m_AudioPacketDuration));

    bool ok = TPCircularBufferInit(&m_RingBuffer,
                                   sizeof(float) *
                                   m_opusConfig->channelCount *
                                   m_opusConfig->samplesPerFrame *
                                   packetsToBuffer);
    if (!ok) return false;

    // Spatial mixer code needs to be able to read from the ring buffer
    m_SpatialAU.setRingBufferPtr(&m_RingBuffer);

    // real length will be larger than requested due to memory page alignment
    m_BufferSize = m_RingBuffer.length;
    DEBUG_TRACE("CoreAudioRenderer ring buffer init, %d packets (%d bytes)", packetsToBuffer, m_BufferSize);

    return true;
}

OSStatus onDeviceOverload(AudioObjectID /*inObjectID*/,
                          UInt32 /*inNumberAddresses*/,
                          const AudioObjectPropertyAddress * /*inAddresses*/,
                          void *inClientData)
{
    CoreAudioRenderer *me = (CoreAudioRenderer *)inClientData;
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "CoreAudioRenderer output device overload");
    me->statsIncDeviceOverload();
    return noErr;
}

OSStatus onAudioNeedsReinit(AudioObjectID /*inObjectID*/,
                            UInt32 /*inNumberAddresses*/,
                            const AudioObjectPropertyAddress * /*inAddresses*/,
                            void *inClientData)
{
    CoreAudioRenderer *me = (CoreAudioRenderer *)inClientData;
    SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION, "CoreAudioRenderer output device had a change, will reinit");
    me->m_needsReinit.store(true);
    return noErr;
}

bool CoreAudioRenderer::initListeners()
{
    // events we care about on our output device

    AudioObjectPropertyAddress addr{kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    OSStatus status = AudioObjectAddPropertyListener(m_OutputDeviceID, &addr, onDeviceOverload, this);
    if (status != noErr) {
        CA_LogError(status, "Failed to add listener for kAudioDeviceProcessorOverload");
        return false;
    }

    addr.mSelector = kAudioDevicePropertyDeviceHasChanged;
    status = AudioObjectAddPropertyListener(m_OutputDeviceID, &addr, onAudioNeedsReinit, this);
    if (status != noErr) {
        CA_LogError(status, "Failed to add listener for kAudioDevicePropertyDeviceHasChanged");
        return false;
    }

    // non-device-specific listeners
    addr.mSelector = kAudioHardwarePropertyServiceRestarted;
    status = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &addr, onAudioNeedsReinit, this);
    if (status != noErr) {
        CA_LogError(status, "Failed to add listener for kAudioHardwarePropertyServiceRestarted");
        return false;
    }

    addr.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
    status = AudioObjectAddPropertyListener(kAudioObjectSystemObject, &addr, onAudioNeedsReinit, this);
    if (status != noErr) {
        CA_LogError(status, "Failed to add listener for kAudioDevicePropertyIOStoppedAbnormally");
        return false;
    }

    return true;
}

void CoreAudioRenderer::deinitListeners()
{
    AudioObjectPropertyAddress addr{kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectRemovePropertyListener(m_OutputDeviceID, &addr, onDeviceOverload, this);

    addr.mSelector = kAudioDevicePropertyDeviceHasChanged;
    AudioObjectRemovePropertyListener(m_OutputDeviceID, &addr, onAudioNeedsReinit, this);

    addr.mSelector = kAudioHardwarePropertyServiceRestarted;
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &addr, onAudioNeedsReinit, this);

    addr.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
    AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &addr, onAudioNeedsReinit, this);
}

bool CoreAudioRenderer::setCallback(AURenderCallback callback)
{
    AURenderCallbackStruct callbackStruct;
    callbackStruct.inputProc = callback;
    callbackStruct.inputProcRefCon = this;

    OSStatus status = AudioUnitSetProperty(m_OutputAU, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callbackStruct, sizeof(callbackStruct));
    if (status != noErr) {
        CA_LogError(status, "Failed to set output render callback");
        return false;
    }

    return true;
}

void CoreAudioRenderer::clearCallback()
{
    AURenderCallbackStruct callbackStruct = {};
    callbackStruct.inputProc = nullptr;
    callbackStruct.inputProcRefCon = nullptr;

    OSStatus status = AudioUnitSetProperty(m_OutputAU, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callbackStruct, sizeof(callbackStruct));
    if (status != noErr) {
        CA_LogError(status, "Error clearing output render callback");
    }
}

void* CoreAudioRenderer::getAudioBuffer(int* size)
{
    // Decode into a guaranteed full-packet staging buffer. Decoding directly
    // into the ring's remaining space could give Opus a partial output frame,
    // which returns OPUS_BUFFER_TOO_SMALL and advances the stream discontinuously.
    const int stagingBytes = (int)(m_DecodeBuffer.size() * sizeof(float));
    if (*size > stagingBytes) {
        *size = stagingBytes;
    }

    return m_DecodeBuffer.data();
}

bool CoreAudioRenderer::submitAudio(int bytesWritten)
{
    // We'll be fully recreated after any changes to the audio device, default output, etc.
    if (m_needsReinit.load()) {
        return false;
    }

    if (bytesWritten == 0) {
        // Nothing to do
        return true;
    }

    const int stagingBytes = (int)(m_DecodeBuffer.size() * sizeof(float));
    if (bytesWritten < 0 || bytesWritten > stagingBytes ||
            m_BytesPerFrame == 0 || (bytesWritten % m_BytesPerFrame) != 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "CoreAudioRenderer received invalid decoded buffer size: %d",
                     bytesWritten);
        return false;
    }

    uint32_t queuedBytes = 0;
    TPCircularBufferTail(&m_RingBuffer, &queuedBytes);
    const int bytesPerMs = m_opusConfig->sampleRate * m_BytesPerFrame / 1000;
    const int queuedAudioMs = bytesPerMs == 0 ? 0 : queuedBytes / bytesPerMs;
    const int totalQueuedAudioMs = LiGetPendingAudioDuration() + queuedAudioMs;

    // Do not punch a hole in an already-starved output queue. We only catch up
    // when the compressed queue exceeds the user's tolerance and local PCM is
    // safely above the rebuffer low-water mark.
    if (totalQueuedAudioMs >= m_JitterBufferMs &&
            queuedBytes >= m_RebufferThresholdBytes) {
        ++m_DropCount;
        m_HadProducerDrop = true;
        return true;
    }

    uint32_t bytesFree = 0;
    void *ringHead = nullptr;
    bool canQueue = false;

    // Match SDL's bounded output queue behavior. Waiting here keeps burst data
    // in the pre-decode queue where the user-selected jitter limit can govern
    // it, rather than permanently inflating playout latency.
    for (int i = 0; i < 100; i++) {
        TPCircularBufferTail(&m_RingBuffer, &queuedBytes);
        ringHead = TPCircularBufferHead(&m_RingBuffer, &bytesFree);
        const int localQueueMs = bytesPerMs == 0 ? 0 : queuedBytes / bytesPerMs;
        if (localQueueMs <= kMaximumLocalQueueMs &&
                ringHead != nullptr && bytesWritten <= (int)bytesFree) {
            canQueue = true;
            break;
        }

        if (m_needsReinit.load()) {
            return false;
        }
        SDL_Delay(1);
    }

    if (!canQueue) {
        ++m_DropCount;
        m_HadProducerDrop = true;
        return true;
    }

    float *decodedSamples = m_DecodeBuffer.data();
    const uint32_t channelCount = m_opusConfig->channelCount;
    const uint32_t framesWritten = bytesWritten / m_BytesPerFrame;

    // If one or more packets were intentionally skipped, blend the next PCM
    // block from the last enqueued sample to avoid a hard waveform step.
    if (m_HadProducerDrop && framesWritten != 0) {
        crossfadeInterleavedFromLast(decodedSamples,
                                     framesWritten,
                                     channelCount,
                                     m_FadeFrames,
                                     m_LastQueuedSamples);
        m_HadProducerDrop = false;
    }

    memcpy(ringHead, decodedSamples, bytesWritten);
    TPCircularBufferProduce(&m_RingBuffer, bytesWritten);

    if (framesWritten != 0) {
        const float *lastFrame = decodedSamples + (framesWritten - 1) * channelCount;
        for (uint32_t channel = 0; channel < channelCount; channel++) {
            m_LastQueuedSamples[channel] = lastFrame[channel];
        }
    }

    // Get buffered audio size
    uint32_t availableBytes = 0;
    TPCircularBufferTail(&m_RingBuffer, &availableBytes);
    m_QueuedAudioSize.store(availableBytes);

    return true;
}

void CoreAudioRenderer::notifyAudioDiscontinuity()
{
    m_HadProducerDrop = true;
}

AUSpatialMixerOutputType CoreAudioRenderer::getSpatialMixerOutputType()
{
#if TARGET_OS_OSX
    // Check if headphones are plugged in.
    uint32_t dataSource{};
    uint32_t size = sizeof(dataSource);

    AudioObjectPropertyAddress addTransType{kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
    OSStatus status = AudioObjectGetPropertyData(m_OutputDeviceID, &addTransType, 0, nullptr, &size, &dataSource);
    if (status != noErr) {
        CA_LogError(status, "Failed to get the transport type of output device");
        return kSpatialMixerOutputType_ExternalSpeakers;
    }

    CA_FourCC(dataSource, m_OutputTransportType);
    DEBUG_TRACE("CoreAudioRenderer output transport type %s", m_OutputTransportType);

    if (dataSource == kAudioDeviceTransportTypeHDMI) {
        dataSource = kIOAudioOutputPortSubTypeExternalSpeaker;
    } else if (dataSource == kAudioDeviceTransportTypeBluetooth || dataSource == kAudioDeviceTransportTypeUSB) {
        dataSource = kIOAudioOutputPortSubTypeHeadphones;
    } else {
        AudioObjectPropertyAddress theAddress{kAudioDevicePropertyDataSource, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementMain};

        status = AudioObjectGetPropertyData(m_OutputDeviceID, &theAddress, 0, nullptr, &size, &dataSource);
        if (status != noErr) {
            CA_LogError(status, "Couldn't determine default audio device type, defaulting to ExternalSpeakers");
            return kSpatialMixerOutputType_ExternalSpeakers;
        }
    }

    CA_FourCC(dataSource, m_OutputDataSource);
    DEBUG_TRACE("CoreAudioRenderer output data source %s", m_OutputDataSource);

    switch (dataSource) {
        case kIOAudioOutputPortSubTypeInternalSpeaker:
            return kSpatialMixerOutputType_BuiltInSpeakers;
            break;

        case kIOAudioOutputPortSubTypeHeadphones:
            return kSpatialMixerOutputType_Headphones;
            break;

        case kIOAudioOutputPortSubTypeExternalSpeaker:
            return kSpatialMixerOutputType_ExternalSpeakers;
            break;

        default:
            return kSpatialMixerOutputType_Headphones;
            break;
    }
#else
    AVAudioSession *audioSession = [AVAudioSession sharedInstance];

    if ([audioSession.currentRoute.outputs count] != 1) {
        return kSpatialMixerOutputType_ExternalSpeakers;
    } else {
        NSString* pType = audioSession.currentRoute.outputs.firstObject.portType;
        if ([pType isEqualToString:AVAudioSessionPortHeadphones] || [pType isEqualToString:AVAudioSessionPortBluetoothA2DP] || [pType isEqualToString:AVAudioSessionPortBluetoothLE] || [pType isEqualToString:AVAudioSessionPortBluetoothHFP]) {
            return kSpatialMixerOutputType_Headphones;
        } else if ([pType isEqualToString:AVAudioSessionPortBuiltInSpeaker]) {
            return kSpatialMixerOutputType_BuiltInSpeakers;
        } else {
            return kSpatialMixerOutputType_ExternalSpeakers;
        }
    }
#endif
}

static void replace_fancy_quote(char *str)
{
    char *pos;
    while ((pos = strstr(str, "\xe2\x80\x99")) != NULL) {
        *pos = '\'';
        memmove(pos + 1, pos + 3, strlen(pos + 3) + 1);
    }
}

void CoreAudioRenderer::setOutputDeviceName(const CFStringRef cfstr)
{
    if (cfstr) {
        CFIndex size = CFStringGetMaximumSizeForEncoding(CFStringGetLength(cfstr), kCFStringEncodingUTF8) + 1;
        char *buffer = (char *)malloc(size);
        CFStringGetCString(cfstr, buffer, size, kCFStringEncodingUTF8);

        // it's very likely we'll get a name like "Andy’s AirPods Pro"
        // with a UTF8 quote, and our overlay font is only ASCII
        replace_fancy_quote(buffer);

        if (m_OutputDeviceName) {
            free(m_OutputDeviceName);
        }

        m_OutputDeviceName = buffer;
    }
}

void CoreAudioRenderer::setHeadTracking(bool enabled)
{
    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "CoreAudioRenderer head tracking set to %d", enabled);
    m_SpatialAU.setHeadTracking(enabled);
    m_needsReinit.store(true);
}
