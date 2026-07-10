// Avoid conflict between AVFoundation and
// libavutil both defining AVMediaType
#define AVMediaType AVMediaType_FFmpeg
#include "vt.h"
#include "pacer/displaylink_source.h"
#undef AVMediaType

#include "imgui.h"
//#include "imgui_impl_osx.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_metal.h"
#include "implot.h"

#include <algorithm>
#include <cmath>
#include "SDL_compat.h"
#include <SDL_syswm.h>
#include <Limelight.h>
#include "streaming/session.h"
#include "streaming/streamutils.h"
#include "framepacing/framequeue.h"
#include "framepacing/framepacer.h"
#include "path.h"
#include "imgui/devui.h"
#include "imgui/gamepadmenu.h"
#include "imgui/imgui_plots.h"
#include "streaming/stats.h"

#import <Cocoa/Cocoa.h>
#import <CoreGraphics/CGDirectDisplayMetal.h>
#import <VideoToolbox/VideoToolbox.h>
#import <AVFoundation/AVFoundation.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

extern "C" {
    #include <libavutil/pixdesc.h>
}

struct CscParams
{
    simd_half3x3 matrix;
    simd_half3 offsets;
};

struct ParamBuffer
{
    CscParams cscParams;
    simd_half2 chromaOffset;
    simd_half1 bitnessScaleFactor;
};

struct Vertex
{
    simd_float4 position;
    simd_float2 texCoord;
};

// tracks in-flight frames and present-to-display latency
// Note that these are in seconds
struct PresentedFrameInfo
{
    uint64_t drawableId;
    int presentMode;
    int pacingMode;
    CFTimeInterval submittedPresentTime;
    CFTimeInterval atTime;
    CFTimeInterval afterMinimumDuration;
    CFTimeInterval vsyncTimestamp;
    CFTimeInterval vsyncDeadline;
    bool late;

    float currentEDR;
    float maxPotentialEDR;
    float maxReferenceEDR;
    float referenceWhite;
    float maxNits;
};

// store any data used by present callbacks in something other than the VTMetalRenderer object.
// If we don't do this, we can crash during destruction.
struct PresentCallbackState
{
    std::mutex pendingFramesMutex;
    std::deque<PresentedFrameInfo> pendingFrames;
    std::atomic<bool> stopping{false};

    std::atomic<double> lastPresented{0.0};
    std::atomic<double> averageGPUTime{1.0 / 240.0};
};

// https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/TripleBuffering.html
// https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work?language=objc
#define MAX_FRAMES_IN_FLIGHT 3

#define MAX_VIDEO_PLANES 3

// CAMetalLayer properties need to be changed on the main thread. The main
// thread also joins the render thread during decoder teardown, so a plain
// dispatch_sync() can deadlock. Queue the self-contained update and allow the
// render thread to abandon its wait once teardown starts. Callers must only
// capture independently retained objects, never the renderer itself.
static bool runOnMainThreadCancellable(dispatch_block_t block)
{
    if ([NSThread isMainThread]) {
        block();
        return true;
    }

    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_main_queue(), ^{
        block();
        dispatch_semaphore_signal(completed);
    });

    while (dispatch_semaphore_wait(completed,
                                   dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_MSEC)) != 0) {
        if (FramePacer::instance().stopping()) {
            return false;
        }
    }

    return true;
}

class VTMetalRenderer;
@interface VTMetalObserver : NSObject

- (id)initWithRenderer:(VTMetalRenderer *)renderer forWindow:(NSWindow *)window;
- (void)stop;

@end

class VTMetalRenderer : public VTBaseRenderer
{
public:
    VTMetalRenderer(bool hwAccel)
        : VTBaseRenderer(RendererType::VTMetal),
          m_HwAccel(hwAccel),
          m_Window(nullptr),
          m_HwContext(nullptr),
          m_MetalLayer(nullptr),
          m_TextureCache(nullptr),
          m_CVMetalTextures{},
          m_CscParamsBuffer(nullptr),
          m_VideoVertexBuffer(nullptr),
          m_OverlayTextures{},
          m_OverlayLock(0),
          m_RenderPassDescriptor(nullptr),
          m_VideoPipelineState(nullptr),
          m_OverlayPipelineState(nullptr),
          m_ShaderLibrary(nullptr),
          m_CommandQueue(nullptr),
          m_CommandBuffer{},
          m_CurrentBuffer(0),
          m_SwMappingTextures{},
          m_MetalView(nullptr),
          m_LastFrameWidth(-1),
          m_LastFrameHeight(-1),
          m_LastDrawableWidth(-1),
          m_LastDrawableHeight(-1),
          m_RenderMaxNits(0.0f),
          m_FrameEncoded(false),
          m_LastSubmittedFramePts(AV_NOPTS_VALUE),
          m_IsFullScreen(false),
          m_ProMotionAllowsVRR(false),
          m_UsePTSForVRR(false),
          m_UseEDR(false),
          m_ShowMetalHUD(false),
          m_MaxPotentialEDR(1.0),
          m_CurrentEDR{1.0},
          m_MaxReferenceEDR(1.0),
          m_ReferenceWhite(203.0f), // 100.0f on displays in reference mode
          m_RequestedPresentMode(StreamingPreferences::PRESENT_AUTO),
          m_MinRefreshInterval(1.0f / 60),
          m_MaxRefreshInterval(1.0f / 60),
          m_SupportsVRR{false},
          m_IsPaused{false},
          m_IsVsync{true},
          m_VsyncTimestamp{0.0},
          m_VsyncDeadline{0.0},
          m_Observer(nil),
          m_IsGPUCapturing(false),
          m_Drawable(nil),
          m_CurrentDrawableID(0)
    {
        StreamingPreferences *prefs = StreamingPreferences::get();
        auto devConfig = DevUISettings::instance().GetConfig();
        int maxFramesInFlight = std::clamp(SDL_min(prefs->vtMetalFramesInFlight, MAX_FRAMES_IN_FLIGHT), 2, 3);
        m_MaxFramesInFlight.store(maxFramesInFlight);
        m_ProMotionAllowsVRR.store(devConfig.proMotionAllowsVRR);
        m_PresentState = std::make_shared<PresentCallbackState>();
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Metal renderer using MaxFramesInFlight=%d", maxFramesInFlight);
   }

    virtual ~VTMetalRenderer() override
    { @autoreleasepool {
        m_PresentState->stopping.store(true);

        if (m_Observer != nil) {
            [m_Observer stop];
            [m_Observer release];
            m_Observer = nil;
        }

        // hide Metal HUD so it doesn't appear over the Qt UI
        showMetalHud(false);

        if (m_HwContext != nullptr) {
            av_buffer_unref(&m_HwContext);
        }

        if (m_CscParamsBuffer != nullptr) {
            [m_CscParamsBuffer release];
        }

        if (m_VideoVertexBuffer != nullptr) {
            [m_VideoVertexBuffer release];
        }

        if (m_RenderPassDescriptor != nullptr) {
            [m_RenderPassDescriptor release];
        }

        if (m_VideoPipelineState != nullptr) {
            [m_VideoPipelineState release];
        }

        for (int i = 0; i < Overlay::OverlayMax; i++) {
            if (m_OverlayTextures[i] != nullptr) {
                [m_OverlayTextures[i] release];
            }
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            for (int j = 0; j < MAX_VIDEO_PLANES; j++) {
                if (m_CVMetalTextures[i][j] != nullptr) {
                    CFRelease(m_CVMetalTextures[i][j]);
                    m_CVMetalTextures[i][j] = nullptr;
                }
                if (m_SwMappingTextures[i][j] != nullptr) {
                    [m_SwMappingTextures[i][j] release];
                }
            }
        }

        if (m_Drawable != nullptr) {
            [m_Drawable release];
            m_Drawable = nullptr;
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (m_CommandBuffer[i] != nullptr) {
                [m_CommandBuffer[i] release];
                m_CommandBuffer[i] = nullptr;
            }
        }

        if (m_OverlayPipelineState != nullptr) {
            [m_OverlayPipelineState release];
        }

        if (m_ShaderLibrary != nullptr) {
            [m_ShaderLibrary release];
        }

        if (m_CommandQueue != nullptr) {
            [m_CommandQueue release];
        }

        if (m_TextureCache != nullptr) {
            CFRelease(m_TextureCache);
        }

        if (m_MetalView != nullptr) {
            SDL_Metal_DestroyView(m_MetalView);
        }

    }}

#ifndef IMGUI_DISABLE
    virtual void ImGui_initBackend() override
    {
        ImGui_ImplMetal_Init(m_MetalLayer.device);
        ImGui_ImplSDL2_InitForMetal(m_Window);
    }

    virtual void ImGui_deinitBackend() override
    {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplSDL2_Shutdown();
    }
#endif

    void showMetalHud(bool visible)
    {
        if (!m_MetalLayer) {
            return;
        }

        if (@available(macOS 13.0, *)) {
            NSDictionary* properties;
            if (visible) {
                properties = @{
                    @"MTL_HUD_OPACITY": @0.8,
                    @"MTL_HUD_DISABLE_MENU_BAR": @0,
                    @"MTL_HUD_ALIGNMENT": @"bottomright",
                    @"MTL_HUD_ELEMENTS": @"device,rosetta,layersize,layerscale,memory,fps,frameinterval,frameintervalgraph,frameintervalhistogram,presentdelay,gputime,thermal,refreshrate,gamemode,client",
                    @"MTL_HUD_SHOW_METRICS_RANGE": @1,
                };
            }
            else {
                properties = @{
                    @"MTL_HUD_OPACITY": @0.0,
                    @"MTL_HUD_DISABLE_MENU_BAR": @1,
                };
            }

            CAMetalLayer* layer = [m_MetalLayer retain];
            properties = [properties retain];
            runOnMainThreadCancellable(^{
                layer.developerHUDProperties = properties;
                [properties release];
                [layer release];
            });
        }
    }

    virtual bool isRenderThreadSupported() override {
        return true;
    }

    virtual bool isVsyncEnabled() override {
        return m_IsVsync.load();
    }

    bool updateVideoRegionSizeForFrame(AVFrame* frame)
    {
        int drawableWidth, drawableHeight;
        SDL_Metal_GetDrawableSize(m_Window, &drawableWidth, &drawableHeight);

        // Check if anything has changed since the last vertex buffer upload
        if (m_VideoVertexBuffer &&
                frame->width == m_LastFrameWidth && frame->height == m_LastFrameHeight &&
                drawableWidth == m_LastDrawableWidth && drawableHeight == m_LastDrawableHeight) {
            // Nothing to do
            return true;
        }

        // Determine the correct scaled size for the video region
        SDL_Rect src, dst;
        src.x = src.y = 0;
        src.w = frame->width;
        src.h = frame->height;
        dst.x = dst.y = 0;
        dst.w = drawableWidth;
        dst.h = drawableHeight;
        StreamUtils::scaleSourceToDestinationSurface(&src, &dst);

        // Convert screen space to normalized device coordinates
        SDL_FRect renderRect;
        StreamUtils::screenSpaceToNormalizedDeviceCoords(&dst, &renderRect, drawableWidth, drawableHeight);

        Vertex verts[] =
        {
            { { renderRect.x, renderRect.y, 0.0f, 1.0f }, { 0.0f, 1.0f } },
            { { renderRect.x, renderRect.y+renderRect.h, 0.0f, 1.0f }, { 0.0f, 0} },
            { { renderRect.x+renderRect.w, renderRect.y, 0.0f, 1.0f }, { 1.0f, 1.0f} },
            { { renderRect.x+renderRect.w, renderRect.y+renderRect.h, 0.0f, 1.0f }, { 1.0f, 0} },
        };

        [m_VideoVertexBuffer release];
        auto bufferOptions = MTLCPUCacheModeWriteCombined | (m_MetalLayer.device.hasUnifiedMemory ? MTLResourceStorageModeShared : MTLResourceStorageModeManaged);
        m_VideoVertexBuffer = [m_MetalLayer.device newBufferWithBytes:verts length:sizeof(verts) options:bufferOptions];
        if (!m_VideoVertexBuffer) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Failed to create video vertex buffer");
            return false;
        }

        m_LastFrameWidth = frame->width;
        m_LastFrameHeight = frame->height;
        m_LastDrawableWidth = drawableWidth;
        m_LastDrawableHeight = drawableHeight;

        return true;
    }

    int getFramePlaneCount(AVFrame* frame)
    {
        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
            return CVPixelBufferGetPlaneCount((CVPixelBufferRef)frame->data[3]);
        }
        else {
            return av_pix_fmt_count_planes((AVPixelFormat)frame->format);
        }
    }

    int getBitnessScaleFactor(AVFrame* frame)
    {
        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
            // VideoToolbox frames never require scaling
            return 1;
        }
        else {
            const AVPixFmtDescriptor* formatDesc = av_pix_fmt_desc_get((AVPixelFormat)frame->format);
            if (!formatDesc) {
                // This shouldn't be possible but handle it anyway
                SDL_assert(formatDesc);
                return 1;
            }

            // This assumes plane 0 is exclusively the Y component
            SDL_assert(formatDesc->comp[0].step == 1 || formatDesc->comp[0].step == 2);
            // Planar 10-bit formats usually store values in the low bits and
            // need scaling after sampling an R16Unorm texture. P010 stores the
            // same values in the high bits, so its component shift already
            // supplies that scaling.
            int shift = (formatDesc->comp[0].step * 8) -
                        formatDesc->comp[0].depth -
                        formatDesc->comp[0].shift;
            shift = std::max(shift, 0);
            return 1 << shift;
        }
    }

    bool updateColorSpaceForFrame(AVFrame* frame)
    {
        bool frameFormatChanged = hasFrameFormatChanged(frame);
        NSData* masteringDisplayInfo = nil;
        NSData* contentLightInfo = nil;
        float metadataMinNits;
        float metadataMaxNits;
        bool overrideNits;
        uint64_t metadataGeneration;
        {
            std::lock_guard<std::mutex> lock(m_HdrMetadataMutex);
            if (!frameFormatChanged && !m_HdrMetadataChanged) {
                return true;
            }

            // Copy Core Foundation metadata while locked so the streaming
            // callback can replace its copy without racing this render pass.
            if (m_MasteringDisplayColorVolume) {
                masteringDisplayInfo =
                    [NSData dataWithData:(__bridge NSData*)m_MasteringDisplayColorVolume];
            }
            if (m_ContentLightLevelInfo) {
                contentLightInfo =
                    [NSData dataWithData:(__bridge NSData*)m_ContentLightLevelInfo];
            }
            metadataMinNits = m_MinNits;
            metadataMaxNits = m_MaxNits;
            overrideNits = m_OverrideNits;
            metadataGeneration = m_HdrMetadataGeneration;
        }

        int colorspace = getFrameColorspace(frame);
        CGColorSpaceRef newColorSpace;
        MTLPixelFormat newPixelFormat;
        CAEDRMetadata* newEDRMetadata = nil;
        const float referenceWhite = m_ReferenceWhite.load();
        ParamBuffer paramBuffer;

        switch (colorspace) {
        case COLORSPACE_REC_709:
            newColorSpace = CGColorSpaceCreateWithName(kCGColorSpaceITUR_709);
            newPixelFormat = MTLPixelFormatBGRA8Unorm;
            break;
        case COLORSPACE_REC_2020:
            newPixelFormat = MTLPixelFormatBGR10A2Unorm;
            if (frame->color_trc == AVCOL_TRC_SMPTE2084) {
                // https://developer.apple.com/documentation/metal/hdr_content/using_color_spaces_to_display_hdr_content
                newColorSpace = CGColorSpaceCreateWithName(kCGColorSpaceITUR_2100_PQ);
            }
            else {
                newColorSpace = CGColorSpaceCreateWithName(kCGColorSpaceITUR_2020);
            }
            break;
        default:
        case COLORSPACE_REC_601:
            newColorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            newPixelFormat = MTLPixelFormatBGRA8Unorm;
            break;
        }

        std::array<float, 9> cscMatrix;
        std::array<float, 3> yuvOffsets;
        std::array<float, 2> chromaOffset;
        getFramePremultipliedCscConstants(frame, cscMatrix, yuvOffsets);
        getFrameChromaCositingOffsets(frame, chromaOffset);

        paramBuffer.cscParams.matrix = simd_matrix(simd_make_half3(cscMatrix[0], cscMatrix[3], cscMatrix[6]),
                                                   simd_make_half3(cscMatrix[1], cscMatrix[4], cscMatrix[7]),
                                                   simd_make_half3(cscMatrix[2], cscMatrix[5], cscMatrix[8]));
        paramBuffer.cscParams.offsets = simd_make_half3(yuvOffsets[0],
                                                        yuvOffsets[1],
                                                        yuvOffsets[2]);
        paramBuffer.chromaOffset = simd_make_half2(chromaOffset[0],
                                                   chromaOffset[1]);

        if (m_UseEDR && frame->color_trc == AVCOL_TRC_SMPTE2084) {
            // EDR requires a linear floating-point pixel format
            newPixelFormat = MTLPixelFormatRGBA16Float;

            // change to linear colorspace
            CFStringRef name;
            switch (colorspace) {
                case COLORSPACE_REC_2020:
                    name = kCGColorSpaceExtendedLinearITUR_2020;
                    break;
                case COLORSPACE_REC_601:
                    name = kCGColorSpaceExtendedLinearSRGB;
                    break;
                case COLORSPACE_REC_709:
                default:
                    name = kCGColorSpaceExtendedLinearSRGB;
                    break;
            }

            CGColorSpaceRelease(newColorSpace);
            newColorSpace = CGColorSpaceCreateWithName(name);

            // MDCV contains min/max values from the host
            // The user can override this if they want to
            if (overrideNits) {
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                            "EDR using user-selected min/max nits %.4f/%.2f, referenceWhite %.2f",
                            metadataMinNits, metadataMaxNits, referenceWhite);
                newEDRMetadata = [CAEDRMetadata HDR10MetadataWithMinLuminance:metadataMinNits
                                                                 maxLuminance:metadataMaxNits
                                                           opticalOutputScale:referenceWhite];
            }
            else if (masteringDisplayInfo != nil) {
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                            "EDR using MasteringDisplayColorVolume from host: min/max nits %.4f/%.2f, referenceWhite %.2f",
                            metadataMinNits, metadataMaxNits, referenceWhite);
                newEDRMetadata = [CAEDRMetadata HDR10MetadataWithDisplayInfo:masteringDisplayInfo
                                                                 contentInfo:contentLightInfo
                                                          opticalOutputScale:referenceWhite];
            }
        }

        // A drawable's format and EDR metadata are fixed when nextDrawable is
        // called. Drop any prefetched drawable before applying changes, even
        // when the change came from stream metadata rather than the DevUI.
        releaseDrawable();

        CAMetalLayer* layer = [m_MetalLayer retain];
        CGColorSpaceRef layerColorSpace = CGColorSpaceRetain(newColorSpace);
        CAEDRMetadata* layerEDRMetadata = [newEDRMetadata retain];
        bool layerUpdated = runOnMainThreadCancellable(^{
            layer.EDRMetadata = layerEDRMetadata;
            layer.pixelFormat = newPixelFormat;
            layer.colorspace = layerColorSpace;

            [layerEDRMetadata release];
            CGColorSpaceRelease(layerColorSpace);
            [layer release];
        });
        if (!layerUpdated) {
            CGColorSpaceRelease(newColorSpace);
            return false;
        }

        paramBuffer.bitnessScaleFactor = getBitnessScaleFactor(frame);

        // The CAMetalLayer retains the CGColorSpace
        CGColorSpaceRelease(newColorSpace);

        // Create the new colorspace parameter buffer for our fragment shader
        [m_CscParamsBuffer release];
        auto bufferOptions = MTLCPUCacheModeWriteCombined | (m_MetalLayer.device.hasUnifiedMemory ? MTLResourceStorageModeShared : MTLResourceStorageModeManaged);
        m_CscParamsBuffer = [m_MetalLayer.device newBufferWithBytes:(void*)&paramBuffer length:sizeof(paramBuffer) options:bufferOptions];
        if (!m_CscParamsBuffer) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Failed to create CSC parameters buffer");
            return false;
        }

        int planes = getFramePlaneCount(frame);
        SDL_assert(planes == 2 || planes == 3);

        NSError* error = nil;
        MTLRenderPipelineDescriptor *pipelineDesc = [[MTLRenderPipelineDescriptor new] autorelease];
        pipelineDesc.vertexFunction = [[m_ShaderLibrary newFunctionWithName:@"vs_draw"] autorelease];

        if (m_MetalLayer.pixelFormat == MTLPixelFormatRGBA16Float) {
            // Linear shader with tonemapping
            pipelineDesc.fragmentFunction = [[m_ShaderLibrary newFunctionWithName:planes == 2 ? @"ps_draw_linear" : @"ps_draw_linear_triplanar"] autorelease];
        }
        else {
            pipelineDesc.fragmentFunction = [[m_ShaderLibrary newFunctionWithName:planes == 2 ? @"ps_draw_biplanar" : @"ps_draw_triplanar"] autorelease];
        }
        pipelineDesc.colorAttachments[0].pixelFormat = m_MetalLayer.pixelFormat;
        [m_VideoPipelineState release];
        m_VideoPipelineState = [m_MetalLayer.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (!m_VideoPipelineState) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Failed to create video pipeline state: %s", error.localizedDescription.UTF8String);
            return false;
        }

        pipelineDesc = [[MTLRenderPipelineDescriptor new] autorelease];
        pipelineDesc.vertexFunction = [[m_ShaderLibrary newFunctionWithName:@"vs_draw"] autorelease];
        pipelineDesc.fragmentFunction = [[m_ShaderLibrary newFunctionWithName:@"ps_draw_rgb"] autorelease];
        pipelineDesc.colorAttachments[0].pixelFormat = m_MetalLayer.pixelFormat;
        pipelineDesc.colorAttachments[0].blendingEnabled = YES;
        pipelineDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        pipelineDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        pipelineDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        [m_OverlayPipelineState release];
        m_OverlayPipelineState = [m_MetalLayer.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (!m_OverlayPipelineState) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Failed to create overlay pipeline state: %s", error.localizedDescription.UTF8String);
            return false;
        }

        m_RenderMaxNits = metadataMaxNits;
        {
            std::lock_guard<std::mutex> lock(m_HdrMetadataMutex);
            if (m_HdrMetadataGeneration == metadataGeneration) {
                m_HdrMetadataChanged = false;
            }
        }
        return true;
    }

    id<MTLTexture> mapPlaneForSoftwareFrame(AVFrame* frame, int planeIndex)
    {
        const AVPixFmtDescriptor* formatDesc = av_pix_fmt_desc_get((AVPixelFormat)frame->format);
        if (!formatDesc) {
            // This shouldn't be possible but handle it anyway
            SDL_assert(formatDesc);
            return nil;
        }

        SDL_assert(planeIndex < MAX_VIDEO_PLANES);

        NSUInteger planeWidth = planeIndex ? AV_CEIL_RSHIFT(frame->width, formatDesc->log2_chroma_w) : frame->width;
        NSUInteger planeHeight = planeIndex ? AV_CEIL_RSHIFT(frame->height, formatDesc->log2_chroma_h) : frame->height;

        int componentCount = 0;
        int packedStep = 0;
        for (int i = 0; i < formatDesc->nb_components; ++i) {
            if (formatDesc->comp[i].plane == planeIndex) {
                ++componentCount;
                packedStep = std::max(packedStep,
                                      static_cast<int>(formatDesc->comp[i].step));
            }
        }

        int bytesPerComponent = componentCount > 0 ? packedStep / componentCount : 0;
        MTLPixelFormat metalFormat;
        if (componentCount == 1 && bytesPerComponent == 1) {
            metalFormat = MTLPixelFormatR8Unorm;
        }
        else if (componentCount == 2 && bytesPerComponent == 1) {
            metalFormat = MTLPixelFormatRG8Unorm;
        }
        else if (componentCount == 1 && bytesPerComponent == 2) {
            metalFormat = MTLPixelFormatR16Unorm;
        }
        else if (componentCount == 2 && bytesPerComponent == 2) {
            metalFormat = MTLPixelFormatRG16Unorm;
        }
        else {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Unhandled software plane layout: %d components, %d-byte step (plane: %d)",
                         componentCount,
                         packedStep,
                         planeIndex);
            SDL_assert(false);
            return nil;
        }

        auto texture = m_SwMappingTextures[m_CurrentBuffer][planeIndex];

        // Recreate when either the dimensions or the underlying software
        // format changes (for example, NV12 to P010 during an HDR transition).
        if (texture && (texture.width != planeWidth ||
                        texture.height != planeHeight ||
                        texture.pixelFormat != metalFormat)) {
            [texture release];
            texture = nil;
            m_SwMappingTextures[m_CurrentBuffer][planeIndex] = nil;
        }

        if (!texture) {
            auto texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metalFormat
                                                                              width:planeWidth
                                                                             height:planeHeight
                                                                          mipmapped:NO];
            texDesc.cpuCacheMode = MTLCPUCacheModeWriteCombined;
            texDesc.storageMode = m_MetalLayer.device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
            texDesc.usage = MTLTextureUsageShaderRead;

            texture = [m_MetalLayer.device newTextureWithDescriptor:texDesc];
            if (!texture) {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                             "Failed to allocate software frame texture");
                return nil;
            }
            m_SwMappingTextures[m_CurrentBuffer][planeIndex] = texture;
        }

        [texture replaceRegion:MTLRegionMake2D(0, 0, planeWidth, planeHeight)
                   mipmapLevel:0
                     withBytes:frame->data[planeIndex]
                   bytesPerRow:frame->linesize[planeIndex]];

        return texture;
    }

    bool createTexturesFromFrame(AVFrame* frame)
    {
        SDL_assert(frame->format == AV_PIX_FMT_VIDEOTOOLBOX);

        CVPixelBufferRef pixBuf = reinterpret_cast<CVPixelBufferRef>(frame->data[3]);
        size_t planes = getFramePlaneCount(frame);

        // Create Metal textures for the planes of the CVPixelBuffer
        for (size_t i = 0; i < planes; i++) {
            MTLPixelFormat fmt;

            switch (CVPixelBufferGetPixelFormatType(pixBuf)) {
            case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            case kCVPixelFormatType_444YpCbCr8BiPlanarVideoRange:
            case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            case kCVPixelFormatType_444YpCbCr8BiPlanarFullRange:
                fmt = (i == 0) ? MTLPixelFormatR8Unorm : MTLPixelFormatRG8Unorm;
                break;

            case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
            case kCVPixelFormatType_444YpCbCr10BiPlanarFullRange:
            case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
            case kCVPixelFormatType_444YpCbCr10BiPlanarVideoRange:
                fmt = (i == 0) ? MTLPixelFormatR16Unorm : MTLPixelFormatRG16Unorm;
                break;

            default:
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                             "Unknown pixel format: %x",
                             CVPixelBufferGetPixelFormatType(pixBuf));
                return false;
            }

            if (m_CVMetalTextures[m_CurrentBuffer][i] != nil) {
                // release previous texture
                CFRelease(m_CVMetalTextures[m_CurrentBuffer][i]);
            }

            CVReturn err = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, m_TextureCache, pixBuf, nullptr, fmt,
                                                                     CVPixelBufferGetWidthOfPlane(pixBuf, i),
                                                                     CVPixelBufferGetHeightOfPlane(pixBuf, i),
                                                                     i,
                                                                     &m_CVMetalTextures[m_CurrentBuffer][i]);
            if (err != kCVReturnSuccess) {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                             "CVMetalTextureCacheCreateTextureFromImage() failed: %d",
                             err);
                return false;
            }
        }

        return true;
    }

    bool testRenderFrame(AVFrame *frame) override
    { @autoreleasepool {
        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
            size_t planes = getFramePlaneCount(frame);
            SDL_assert(planes <= MAX_VIDEO_PLANES);

            // Test that we can actually create Metal textures from the CVPixelBufferRef
            if (!createTexturesFromFrame(frame)) {
                return false;
            }
        }
        else {
            // Mapping software frames should always work
        }

        return true;
    }}

    // Caller frees frame after we return
    virtual bool renderFrameIntoDrawable(AVFrame* frame, id<CAMetalDrawable> drawable)
    { @autoreleasepool {
        size_t planes = getFramePlaneCount(frame);
        SDL_assert(planes <= MAX_VIDEO_PLANES);

        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
            if (!createTexturesFromFrame(frame)) {
                return false;
            }
        }

        m_RenderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        auto commandBuffer = getCommandBuffer();
        if (!commandBuffer) {
            m_RenderPassDescriptor.colorAttachments[0].texture = nil;
            return false;
        }

        auto renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:m_RenderPassDescriptor];
        if (!renderEncoder) {
            m_RenderPassDescriptor.colorAttachments[0].texture = nil;
            return false;
        }

        // Bind textures and buffers then draw the video region
        [renderEncoder setRenderPipelineState:m_VideoPipelineState];
        if (frame->format == AV_PIX_FMT_VIDEOTOOLBOX) {
            for (size_t i = 0; i < planes; i++) {
                [renderEncoder setFragmentTexture:CVMetalTextureGetTexture(m_CVMetalTextures[m_CurrentBuffer][i]) atIndex:i];
            }
        }
        else {
            for (size_t i = 0; i < planes; i++) {
                id<MTLTexture> texture = mapPlaneForSoftwareFrame(frame, i);
                if (!texture) {
                    [renderEncoder endEncoding];
                    m_RenderPassDescriptor.colorAttachments[0].texture = nil;
                    return false;
                }
                [renderEncoder setFragmentTexture:texture atIndex:i];
            }
        }
        [renderEncoder setVertexBuffer:m_VideoVertexBuffer offset:0 atIndex:0];
        if (m_MetalLayer.pixelFormat == MTLPixelFormatRGBA16Float) {
            float currentEDR = m_CurrentEDR.load();
            float referenceWhite = m_ReferenceWhite.load();
            [renderEncoder setFragmentBytes:&currentEDR length:sizeof(float) atIndex:1];
            [renderEncoder setFragmentBytes:&referenceWhite length:sizeof(float) atIndex:2];
            [renderEncoder setFragmentBytes:&m_RenderMaxNits length:sizeof(float) atIndex:3];
        }
        [renderEncoder setFragmentBuffer:m_CscParamsBuffer offset:0 atIndex:0];
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];

        // Now draw any overlays that are enabled
        for (int i = 0; i < Overlay::OverlayMax; i++) {
            id<MTLTexture> overlayTexture = nullptr;

            // Try to acquire a reference on the overlay texture
            SDL_AtomicLock(&m_OverlayLock);
            overlayTexture = [m_OverlayTextures[i] retain];
            SDL_AtomicUnlock(&m_OverlayLock);

            if (overlayTexture) {
                SDL_FRect renderRect = {};
                if (i == Overlay::OverlayStatusUpdate) {
                    // Bottom Left
                    renderRect.x = 0;
                    renderRect.y = 0;
                }
                else if (i == Overlay::OverlayDebug) {
                    // Top left
                    renderRect.x = 0;
                    renderRect.y = m_LastDrawableHeight - overlayTexture.height;
                }

                renderRect.w = overlayTexture.width;
                renderRect.h = overlayTexture.height;

                // Convert screen space to normalized device coordinates
                StreamUtils::screenSpaceToNormalizedDeviceCoords(&renderRect, m_LastDrawableWidth, m_LastDrawableHeight);

                Vertex verts[] =
                {
                    { { renderRect.x, renderRect.y, 0.0f, 1.0f }, { 0.0f, 1.0f } },
                    { { renderRect.x, renderRect.y+renderRect.h, 0.0f, 1.0f }, { 0.0f, 0} },
                    { { renderRect.x+renderRect.w, renderRect.y, 0.0f, 1.0f }, { 1.0f, 1.0f} },
                    { { renderRect.x+renderRect.w, renderRect.y+renderRect.h, 0.0f, 1.0f }, { 1.0f, 0} },
                };

                [renderEncoder setRenderPipelineState:m_OverlayPipelineState];
                [renderEncoder setFragmentTexture:overlayTexture atIndex:0];
                [renderEncoder setVertexBytes:verts length:sizeof(verts) atIndex:0];
                [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];

                [overlayTexture release];
            }
        }

#ifndef IMGUI_DISABLE
        // avoid a crash at shutdown due to ImGui calling SDL
        // TODO; refactor this out to a parent class
        if (!FramePacer::instance().stopping()) {
            ImGui_ImplMetal_NewFrame(m_RenderPassDescriptor);
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            // Dockspace is cool but overkill
            //ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

            DisplayOutputFormat outputFormat =
                m_MetalLayer.pixelFormat == MTLPixelFormatBGR10A2Unorm ? OUTPUT_IS_PQ
                : m_MetalLayer.pixelFormat == MTLPixelFormatRGBA16Float ? OUTPUT_IS_LINEAR
                : OUTPUT_IS_SDR;
            DevUIColors.InitColors(outputFormat);

            Stats::instance().RenderGraphs();
            DevUISettings::instance().Render();
            GamepadMenu::instance().Render();

            ImGui::EndFrame();
            ImGui::Render();
            ImDrawData* draw_data = ImGui::GetDrawData();
            ImGui_ImplMetal_RenderDrawData(draw_data, commandBuffer, renderEncoder);
        }
#endif

        [renderEncoder endEncoding];
        m_RenderPassDescriptor.colorAttachments[0].texture = nil;
        return true;
    }}

    virtual void presentFrame(AVFrame* frame, uint64_t targetQpc) override
    { @autoreleasepool {
        if (!m_FrameEncoded || m_IsPaused.load()) {
            discardPendingFrame();
            return;
        }

        auto commandBuffer = m_CommandBuffer[m_CurrentBuffer];
        auto drawable = m_Drawable;
        if (!commandBuffer || !drawable) {
            discardPendingFrame();
            return;
        }
        m_FrameEncoded = false;

        // Track the last frame actually submitted for presentation. Decoder
        // arrival order is insufficient here because queued frames can drop,
        // and display-locked pacing can present the same frame more than once.
        int64_t prevPts = m_LastSubmittedFramePts;
        bool isNewFrame = prevPts == AV_NOPTS_VALUE || frame->pts != prevPts;
        if (isNewFrame) {
            m_LastSubmittedFramePts = frame->pts;
        }

        // Warning: callbacks must only access the shared state captured here,
        // never VTMetalRenderer members. This lets committed GPU work finish
        // safely without blocking renderer destruction.
        auto state = m_PresentState;
        [drawable addPresentedHandler:^(id<MTLDrawable> d) {
            if (state->stopping.load()) {
                return;
            }

            // values we'll pass to DevUI
            double presentDelayMs = 0.0;
            double presentIntervalMs = 0.0;

            // Graph frametime. Note that d.presentedTime can be 0.0 if the frame was missed for some reason.
            CFTimeInterval lastPresented = (CFTimeInterval)state->lastPresented.load();
            FQLog("[%f] dID %lu d.presentedTime %f, lastPresented %f, frametime %.3fms",
                CACurrentMediaTime(), (unsigned long)d.drawableID, d.presentedTime, lastPresented, (d.presentedTime - lastPresented) * 1000.0);

            // frametime, not counting duplicates in DL mode
            if (isNewFrame) {
                if (lastPresented > 0.0 && d.presentedTime > 0.0) {
                    presentIntervalMs = (d.presentedTime - lastPresented) * 1000.0;
                    ImGuiPlots::instance().observeFloat(PLOT_FRAMETIME, static_cast<float>(presentIntervalMs));
                }
                state->lastPresented.store((double)d.presentedTime);
            }

            // present-to-display latency
            {
                std::lock_guard<std::mutex> lock(state->pendingFramesMutex);
                for (auto it = state->pendingFrames.begin(); it != state->pendingFrames.end(); ++it) {
                    if (it->drawableId == d.drawableID) {
                        const double submittedPresentTime = it->submittedPresentTime;
                        int presentMode = it->presentMode;

                        if (d.presentedTime > 0.0) {
                            double presentDelay = d.presentedTime - submittedPresentTime;
                            Stats::instance().SubmitPresentTimeUs(
                                static_cast<uint64_t>(presentDelay * 1000000), presentMode);
                            presentDelayMs = presentDelay * 1000.0;
                            ImGuiPlots::instance().observeFloat(PLOT_PRESENT_DELAY, static_cast<float>(presentDelayMs));
                        }

                        state->pendingFrames.erase(it);
                        break;
                    }
                }
            }

        #ifndef IMGUI_DISABLE
            DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
                if (presentDelayMs > 0.0) {
                    metrics.presentDelayMs.add(presentDelayMs);
                    metrics.presentIntervalMs.add(presentIntervalMs);
                }
            });
        #endif
        }];

        auto recordPresentedFrame = [state](const PresentedFrameInfo& pfi) {
            std::lock_guard<std::mutex> lock(state->pendingFramesMutex);
            state->pendingFrames.push_back(pfi);

        #ifndef IMGUI_DISABLE
            // pass this info for display in the dev UI
            // More stats are set when the frame is presented in addPresentedHandler
            DevUISettings::instance().UpdateMetrics([&](DevUIMetrics& metrics) {
                metrics.presentMode = pfi.presentMode;
                metrics.presentAtTime = pfi.atTime;
                metrics.presentAMDHistory.add(pfi.afterMinimumDuration);
                if (pfi.submittedPresentTime > pfi.vsyncDeadline) {
                    // we missed the current deadline
                    metrics.presentLateMs.add((pfi.submittedPresentTime - pfi.vsyncDeadline) * 1000.0);
                }
                if (pfi.pacingMode == StreamingPreferences::FRAME_PACING_DISPLAY_LOCKED && pfi.late) {
                    // alternate check for lateness
                    metrics.displayLockedMissed++;
                }
                if (pfi.currentEDR > 1.0) {
                    metrics.currentEDR = pfi.currentEDR;
                    metrics.maxPotentialEDR = pfi.maxPotentialEDR;
                    metrics.maxReferenceEDR = pfi.maxReferenceEDR;
                    metrics.referenceWhite = pfi.referenceWhite;
                    metrics.maxNits = pfi.maxNits;
                }
            });
        #endif
        };

        // track how well we're running within the frame
        CFTimeInterval vsyncTimestamp = m_VsyncTimestamp.load();
        CFTimeInterval vsyncDeadline = m_VsyncDeadline.load();

        auto pfi = PresentedFrameInfo{
            .drawableId     = m_CurrentDrawableID,
            .pacingMode     = FramePacer::instance().GetPacingMode(),
            .atTime         = 0.0,
            .vsyncTimestamp = vsyncTimestamp,
            .vsyncDeadline  = vsyncDeadline
        };
        pfi.submittedPresentTime = CACurrentMediaTime();

        if (m_UseEDR) {
            pfi.currentEDR = m_CurrentEDR.load();
            pfi.maxPotentialEDR = m_MaxPotentialEDR.load();
            pfi.maxReferenceEDR = m_MaxReferenceEDR.load();
            pfi.referenceWhite = m_ReferenceWhite.load();
            pfi.maxNits = m_RenderMaxNits;
        }

        // prefer the user's choice from DevUI, or use auto-detection
        pfi.presentMode = m_RequestedPresentMode;
        if (pfi.presentMode == StreamingPreferences::PRESENT_AUTO) {
            if (m_SupportsVRR.load()) {
                // display supports VRR
                pfi.presentMode = StreamingPreferences::PRESENT_VRR;
                // VRR should always use Immediate pacing
                pfi.pacingMode = StreamingPreferences::FRAME_PACING_IMMEDIATE;
                FramePacer::instance().SetPacingMode(pfi.pacingMode);
            }
            else if (m_IsVsync.load()) {
                // vsync enabled
                pfi.presentMode = StreamingPreferences::PRESENT_FIXED;
            }
            else {
                // vsync disabled
                pfi.presentMode = StreamingPreferences::PRESENT_NO_VSYNC;
            }
        }

        // If VRR is manually selected, only allow it in fullscreen
        if (pfi.presentMode == StreamingPreferences::PRESENT_VRR && !m_IsFullScreen.load()) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "VRR presentMode selected but app is not fullscreen, using Fixed");
            pfi.presentMode = StreamingPreferences::PRESENT_FIXED;
        }

        if (pfi.presentMode == StreamingPreferences::PRESENT_VRR) {
            // In VRR mode we schedule based on how long the previous
            // frame should be displayed. The default uses a low value
            // that covers the average GPU time per frame.
            CFTimeInterval duration = 0.0;
            double avgGPUTime = state->averageGPUTime.load();
            CFTimeInterval maxRefreshInterval = m_MaxRefreshInterval.load();

            // Optionally we can use the host's timestamps to schedule
            // each frame for as long as the last actually submitted frame was
            // displayed. Unsigned subtraction handles the 32-bit RTP wrap.
            if (m_UsePTSForVRR && frame->pts != AV_NOPTS_VALUE &&
                    prevPts != AV_NOPTS_VALUE && isNewFrame) {
                uint32_t deltaPts = static_cast<uint32_t>(frame->pts) -
                                    static_cast<uint32_t>(prevPts);
                // One second is already below the useful range for streaming
                // presentation. Treat larger deltas as a discontinuity rather
                // than a 32-bit wrap so a timestamp reset cannot stall VRR.
                if (deltaPts > 0 && deltaPts <= 90000) {
                    double delta = static_cast<double>(deltaPts) / 90000.0;

                    // We could clamp to the monitor's reported range, but the
                    // display can use low-framerate compensation below it.
                    duration = std::clamp(delta, avgGPUTime, 1.0);

                    FQLog("[%f] VRR mode pts %.3fs, prevPts %.3fs, delta %.3fs, bounds %.3f/%.3f, present afterMinimumDuration:%.3f ms",
                        CACurrentMediaTime(), frame->pts / 90000.0, prevPts / 90000.0, delta, avgGPUTime * 1000.0, maxRefreshInterval * 1000.0, duration * 1000.0);
                }
            }
            if (duration <= 0.0) {
                // Apple recommends using a min duration using the average GPU time when rendering for VRR.
                duration = std::clamp(avgGPUTime, 0.0, maxRefreshInterval);

                FQLog("[%f] VRR mode %.3fs, bounds %.3f/%.3f, present afterMinimumDuration:%.3f ms",
                    CACurrentMediaTime(), frame->pts / 90000.0, avgGPUTime * 1000.0, maxRefreshInterval * 1000.0, duration * 1000.0);
            }

            pfi.afterMinimumDuration = duration;
        }
        else if (pfi.presentMode == StreamingPreferences::PRESENT_FIXED) {
            // targetQpc and the display-link deadline both identify the next
            // desired vblank. Do not add an unconditional refresh interval:
            // doing so adds 16.7 ms at 60 Hz (8.3 ms at 120 Hz). Roll forward
            // only when there is no longer enough lead time for the GPU.
            CFTimeInterval interval = vsyncDeadline - vsyncTimestamp;
            if (interval > 0.0 && interval < 1.0) {
                CFTimeInterval targetTime = targetQpc ?
                    QpcToMs(targetQpc) / 1000.0 : vsyncDeadline;
                CFTimeInterval earliestUsefulTarget =
                    pfi.submittedPresentTime + std::max(state->averageGPUTime.load(), 0.0005);

                while (targetTime > 0.0 && targetTime <= earliestUsefulTarget) {
                    targetTime += interval;
                    pfi.late = true;
                }

                pfi.atTime = targetTime;
            }
        }

        // Metal's command-buffer presentation methods are themselves invoked
        // when the command buffer is scheduled. Registering presentation here
        // avoids a second asynchronous scheduled-handler hop and guarantees
        // that a committed buffer still presents during renderer teardown.
        switch (pfi.presentMode) {
            case StreamingPreferences::PRESENT_VRR:
                [commandBuffer presentDrawable:drawable
                           afterMinimumDuration:pfi.afterMinimumDuration];
                break;

            case StreamingPreferences::PRESENT_NO_VSYNC:
                [commandBuffer presentDrawable:drawable];
                break;

            default:
            case StreamingPreferences::PRESENT_FIXED:
                if (pfi.atTime > 0.0) {
                    [commandBuffer presentDrawable:drawable atTime:pfi.atTime];
                }
                else {
                    [commandBuffer presentDrawable:drawable];
                }
                break;
        }
        recordPresentedFrame(pfi);

        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            if (state->stopping.load()) {
                return;
            }

            // track GPU time
            if (cb.status == MTLCommandBufferStatusCompleted &&
                    cb.GPUStartTime > 0.0 && cb.GPUEndTime >= cb.GPUStartTime) {
                const CFTimeInterval GPUTime = cb.GPUEndTime - cb.GPUStartTime;
                const double alpha = 0.25;
                double avgGPUTime = (GPUTime * alpha) +
                    (state->averageGPUTime.load() * (1.0 - alpha));
                state->averageGPUTime.store(avgGPUTime);
            }
        }];

        [commandBuffer commit];
        [m_CommandBuffer[m_CurrentBuffer] release];
        m_CommandBuffer[m_CurrentBuffer] = nil;

        // also check for updated DevUI settings
        applyDevUIConfig();

        // Force a new drawable for the next frame.
        // We expect this to block, this works better for
        // frame pacing than a semaphore according to RetroArch
        if (!m_IsPaused.load() && !FramePacer::instance().stopping()) {
            nextDrawable(true);
        }
    }}

    // Caller frees frame after we return
    virtual void renderFrame(AVFrame* frame) override
    { @autoreleasepool {
        m_FrameEncoded = false;

        if (m_IsPaused.load()) {
            // we're paused due to being hidden or off screen,
            // so release any prefetched drawable and throw away the frame.
            discardPendingFrame();
            return;
        }

        m_CurrentBuffer = (m_CurrentBuffer + 1) % m_MaxFramesInFlight.load();

        // Handle changes to the frame's colorspace from last time we rendered
        if (!updateColorSpaceForFrame(frame)) {
            // Trigger the main thread to recreate the decoder
            SDL_Event event;
            event.type = SDL_RENDER_DEVICE_RESET;
            SDL_PushEvent(&event);
            discardPendingFrame();
            return;
        }

        // Handle changes to the video size or drawable size
        if (!updateVideoRegionSizeForFrame(frame)) {
            // Trigger the main thread to recreate the decoder
            SDL_Event event;
            event.type = SDL_RENDER_DEVICE_RESET;
            SDL_PushEvent(&event);
            discardPendingFrame();
            return;
        }

        // Render to the next drawable
        id<CAMetalDrawable> drawable = nextDrawable();
        if (drawable == nullptr) {
            discardPendingFrame();
            return;
        }

        m_FrameEncoded = renderFrameIntoDrawable(frame, drawable);
        if (!m_FrameEncoded) {
            discardPendingFrame();
        }
    }}

    // Returns a retained reference to the Metal device currently driving the
    // window's display. On Intel multi-GPU systems this avoids rendering on the
    // low-power GPU only to copy every frame to a dGPU/eGPU for presentation.
    id<MTLDevice> copyWindowDisplayMetalDevice()
    {
        // AppKit window/screen access is main-thread only. Test-only renderers
        // may be initialized from a worker and should use the normal fallback.
        if (![NSThread isMainThread] || !m_Window) {
            return nil;
        }

        NSWindow* window = getNSWindow();
        NSScreen* screen = window.screen;
        NSNumber* screenNumber = screen.deviceDescription[@"NSScreenNumber"];
        if (!screenNumber) {
            return nil;
        }

        CGDirectDisplayID displayId =
            static_cast<CGDirectDisplayID>(screenNumber.unsignedIntValue);
        return CGDirectDisplayCopyCurrentMetalDevice(displayId);
    }

    id<MTLDevice> getMetalDevice() {
        StreamingPreferences *prefs = StreamingPreferences::get();
        if (prefs->renderer == StreamingPreferences::RENDERER_AVSAMPLEBUFFER || qgetenv("VT_FORCE_METAL") == "0") {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Avoiding Metal renderer due to VT_FORCE_METAL=0 override.");
            return nullptr;
        }

        id<MTLDevice> displayDevice = copyWindowDisplayMetalDevice();
        if (displayDevice) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Using Metal device that drives the current display: %s",
                        displayDevice.name.UTF8String);
            return [displayDevice autorelease];
        }

        NSArray<id<MTLDevice>> *devices = [MTLCopyAllDevices() autorelease];
        if (devices.count == 0) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "No Metal device found!");
            return nullptr;
        }

        // First, try to find a low power (Intel) or unified memory (Apple Silicon) GPU
        for (id<MTLDevice> device in devices) {
            // choose iGPU on Intel Macs
            if (device.isLowPower || device.hasUnifiedMemory) {
                return device;
            }
        }

        // Next, we'll just try to pick something internal to the system
        for (id<MTLDevice> device in devices) {
            if (!device.isRemovable) {
                return device;
            }
        }

        // Use the system-default device
        return [MTLCreateSystemDefaultDevice() autorelease];
    }

    virtual bool initialize(PDECODER_PARAMETERS params) override
    { @autoreleasepool {
        int err;
        m_Window = params->window;
        id<MTLDevice> device = getMetalDevice();
        if (!device) {
            m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
            return false;
        }

        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Selected Metal device: %s",
                    device.name.UTF8String);

        if (m_HwAccel && !checkDecoderCapabilities(device, params)) {
            return false;
        }

        err = av_hwdevice_ctx_create(&m_HwContext,
                                     AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                                     nullptr,
                                     nullptr,
                                     0);
        if (err < 0) {
            SDL_LogWarn(SDL_LOG_CATEGORY_APPLICATION,
                        "av_hwdevice_ctx_create() failed for VT decoder: %d",
                        err);
            m_InitFailureReason = InitFailureReason::NoSoftwareSupport;
            return false;
        }

        // Create the Metal texture cache for our CVPixelBuffers
        CFStringRef keys[] = { kCVMetalTextureUsage };
        NSUInteger usage = MTLTextureUsageShaderRead;
        CFNumberRef usageNumber = CFNumberCreate(kCFAllocatorDefault, kCFNumberNSIntegerType, &usage);
        const void* values[] = { usageNumber };
        auto cacheAttributes = CFDictionaryCreate(kCFAllocatorDefault, (const void**)keys, values, 1, nullptr, nullptr);
        err = CVMetalTextureCacheCreate(kCFAllocatorDefault, cacheAttributes, device, nullptr, &m_TextureCache);
        CFRelease(cacheAttributes);
        CFRelease(usageNumber);

        if (err != kCVReturnSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "CVMetalTextureCacheCreate() failed: %d",
                         err);
            return false;
        }

        // Compile our shaders
        NSError* error = nil;
        QString shaderSource = QString::fromUtf8(Path::readDataFile("vt_renderer.metal"));
        m_ShaderLibrary = [device newLibraryWithSource:shaderSource.toNSString() options:nullptr error:&error];
        if (!m_ShaderLibrary) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "Failed to compile shaders: %s", error.localizedDescription.UTF8String);
            return false;
        }

        // Create a command queue for submission
        m_CommandQueue = [device newCommandQueue];

        // we'll reuse one renderPassDescriptor by changing its texture
        m_RenderPassDescriptor = [[MTLRenderPassDescriptor alloc] init];
        m_RenderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        m_RenderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        m_RenderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

        // Add the Metal view to the window if we're not in test-only mode
        //
        // NB: Test-only renderers may be created on a non-main thread, so
        // we don't want to touch the view hierarchy in that context.
        if (!params->testOnly) {
            m_MetalView = SDL_Metal_CreateView(m_Window);
            if (!m_MetalView) {
                SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                             "SDL_Metal_CreateView() failed: %s",
                             SDL_GetError());
                return false;
            }

            m_MetalLayer = (CAMetalLayer*)SDL_Metal_GetLayer(m_MetalView);

            // Choose a device
            m_MetalLayer.device = device;
            m_MetalLayer.maximumDrawableCount = std::clamp(m_MaxFramesInFlight.load(), 2, 3);
            m_MetalLayer.allowsNextDrawableTimeout = YES;
            m_MetalLayer.opaque = YES;

            // Allow EDR content if we're streaming in a 10-bit format
            m_MetalLayer.wantsExtendedDynamicRangeContent = !!(params->videoFormat & VIDEO_FORMAT_MASK_10BIT);

            // check fullscreen state, VRR, and refresh rate
            refreshWindowMetadata();

            // I would completely remove the ability to run this renderer without vsync
            // but someone may want it for something.
            m_MetalLayer.displaySyncEnabled = YES;
            m_IsVsync.store(true);
            if (m_IsFullScreen.load()) {
                if (!params->enableVsync) {
                    // Allow v-sync disabled only in fullscreen
                    m_MetalLayer.displaySyncEnabled = NO;
                    m_IsVsync.store(false);
                    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                                "V-sync disabled per request in fullscreen");
                }
            }
            else {
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                            "V-sync enforced when running in a window");
            }
        }

        return true;
    }}

    virtual void notifyOverlayUpdated(Overlay::OverlayType type) override
    { @autoreleasepool {
        SDL_Surface* newSurface = Session::get()->getOverlayManager().getUpdatedOverlaySurface(type);
        bool overlayEnabled = Session::get()->getOverlayManager().isOverlayEnabled(type);
        if (newSurface == nullptr && overlayEnabled) {
            // The overlay is enabled and there is no new surface. Leave the old texture alone.
            return;
        }

        SDL_AtomicLock(&m_OverlayLock);
        auto oldTexture = m_OverlayTextures[type];
        m_OverlayTextures[type] = nullptr;
        SDL_AtomicUnlock(&m_OverlayLock);
        [oldTexture release];

        // If the overlay is disabled, we're done
        if (!overlayEnabled) {
            SDL_FreeSurface(newSurface);
            return;
        }

        // Create a texture to hold our pixel data
        SDL_assert(!SDL_MUSTLOCK(newSurface));
        SDL_assert(newSurface->format->format == SDL_PIXELFORMAT_ARGB8888);
        auto texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                          width:newSurface->w
                                                                         height:newSurface->h
                                                                      mipmapped:NO];
        texDesc.cpuCacheMode = MTLCPUCacheModeWriteCombined;
        texDesc.storageMode = m_MetalLayer.device.hasUnifiedMemory ? MTLStorageModeShared : MTLStorageModeManaged;
        texDesc.usage = MTLTextureUsageShaderRead;
        auto newTexture = [m_MetalLayer.device newTextureWithDescriptor:texDesc];

        // Load the pixel data into the new texture
        [newTexture replaceRegion:MTLRegionMake2D(0, 0, newSurface->w, newSurface->h)
                      mipmapLevel:0
                        withBytes:newSurface->pixels
                      bytesPerRow:newSurface->pitch];

        // The surface is no longer required
        SDL_FreeSurface(newSurface);
        newSurface = nullptr;

        SDL_AtomicLock(&m_OverlayLock);
        m_OverlayTextures[type] = newTexture;
        SDL_AtomicUnlock(&m_OverlayLock);
    }}

    virtual void notifyVsyncTimestamps(double timestamp, double deadline) override
    {
        m_VsyncTimestamp.store(timestamp);
        m_VsyncDeadline.store(deadline);
    }

    virtual bool prepareDecoderContext(AVCodecContext* context, AVDictionary**) override
    {
        if (m_HwAccel) {
            context->hw_device_ctx = av_buffer_ref(m_HwContext);
        }

        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Using Metal renderer with %s decoding",
                    m_HwAccel ? "hardware" : "software");

        return true;
    }

    int getDecoderColorspace() override
    {
        // macOS seems to handle Rec 601 best
        return COLORSPACE_REC_601;
    }

    int getDecoderCapabilities() override
    {
        return CAPABILITY_REFERENCE_FRAME_INVALIDATION_HEVC |
               CAPABILITY_REFERENCE_FRAME_INVALIDATION_AV1;
    }

    int getRendererAttributes() override
    {
        return RENDERER_ATTRIBUTE_HDR_SUPPORT;
    }

    bool isPixelFormatSupported(int videoFormat, AVPixelFormat pixelFormat) override
    {
        if (m_HwAccel) {
            return pixelFormat == AV_PIX_FMT_VIDEOTOOLBOX;
        }
        else {
            if (pixelFormat == AV_PIX_FMT_VIDEOTOOLBOX) {
                // VideoToolbox frames are always supported
                return true;
            }
            else {
                // Otherwise it's supported if we can map it
                const int expectedPixelDepth = (videoFormat & VIDEO_FORMAT_MASK_10BIT) ? 10 : 8;
                const int expectedLog2ChromaW = (videoFormat & VIDEO_FORMAT_MASK_YUV444) ? 0 : 1;
                const int expectedLog2ChromaH = (videoFormat & VIDEO_FORMAT_MASK_YUV444) ? 0 : 1;

                const AVPixFmtDescriptor* formatDesc = av_pix_fmt_desc_get(pixelFormat);
                if (!formatDesc) {
                    // This shouldn't be possible but handle it anyway
                    SDL_assert(formatDesc);
                    return false;
                }

                int planes = av_pix_fmt_count_planes(pixelFormat);
                return (planes == 2 || planes == 3) &&
                       formatDesc->comp[0].depth == expectedPixelDepth &&
                       formatDesc->log2_chroma_w == expectedLog2ChromaW &&
                       formatDesc->log2_chroma_h == expectedLog2ChromaH;
            }
        }
    }

    void refreshWindowMetadata()
    {
        // Get the current window from SDL if it wasn't passed in
        NSWindow* nswindow = getNSWindow();
        if (nswindow) {
            NSScreen* screen = [nswindow screen];
            if (!screen) {
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                            "Output display: not found");
                return;
            }

            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Display change: %s", [screen.localizedName UTF8String]);
            NSDictionary *deviceDescription = screen.deviceDescription;
            [deviceDescription enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL*) {
                NSString *keyString = [key description];
                NSString *valueString = [value description];
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "%s: %s",
                            keyString.UTF8String, valueString.UTF8String);
            }];

            // SDL's display change detection will call notifyWindowChanged() but there are some Apple-specific
            // notifications we need to watch out for, such as moving off screen or into the background.
            // We still have to decode everything anyway, but DisplayLink changes framerates when the window is not
            // visible, and it's better to pause in that situation.
            if (m_Observer != nil) {
                [m_Observer stop];
                [m_Observer release];
                m_Observer = nil;
            }
            m_Observer = [[VTMetalObserver alloc] initWithRenderer:this forWindow:nswindow];

            bool isVRR = false;
            bool isProMotion = false;
            CFTimeInterval displayUpdateGranularity = 0.0;
            CFTimeInterval minRefreshInterval;
            CFTimeInterval maxRefreshInterval;
            if (@available(macOS 12.0, *)) {
                minRefreshInterval = screen.minimumRefreshInterval; // highest Hz
                maxRefreshInterval = screen.maximumRefreshInterval; // lowest Hz
                displayUpdateGranularity = screen.displayUpdateGranularity;

                isVRR = std::abs(minRefreshInterval - maxRefreshInterval) > 0.000001;
                // Fixed-rate displays also report a non-zero granularity. The
                // discrete VRR range plus granularity is what identifies the
                // ProMotion-style behavior relevant to our presentation mode.
                isProMotion = isVRR && displayUpdateGranularity > 0.0;
            }
            else {
                RefreshRateRational refreshRate =
                    StreamUtils::getDisplayRefreshRateRational(m_Window);
                double hz = refreshRate.hz > 0.0 ? refreshRate.hz : 60.0;
                minRefreshInterval = maxRefreshInterval = 1.0 / hz;
            }
            if (minRefreshInterval <= 0.0 || maxRefreshInterval <= 0.0) {
                RefreshRateRational refreshRate =
                    StreamUtils::getDisplayRefreshRateRational(m_Window);
                double hz = refreshRate.hz > 0.0 ? refreshRate.hz : 60.0;
                minRefreshInterval = maxRefreshInterval = 1.0 / hz;
                isVRR = isProMotion = false;
            }
            m_MinRefreshInterval.store(minRefreshInterval);
            m_MaxRefreshInterval.store(maxRefreshInterval);

            bool isFullScreen =
                (nswindow.styleMask & NSWindowStyleMaskFullScreen) == NSWindowStyleMaskFullScreen;
            m_IsFullScreen.store(isFullScreen);
            if (isProMotion) {
                if (m_ProMotionAllowsVRR.load()) {
                    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                                "ProMotion display detected, but allowing VRR from %.0f-%.0f Hz",
                                1.0 / maxRefreshInterval, 1.0 / minRefreshInterval);
                    isVRR = true;
                }
                else {
                    // ProMotion displays like MacBook Pro are detected as VRR but behave
                    // badly since they can only operate at 120hz, 60hz, and a few lower rates.
                    // This should also handle the low power use case where it will run at 60hz.
                    if (isFullScreen) {
                        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                                    "ProMotion display detected (displayUpdateGranularity %.3fms), treating as a fixed %.2f Hz display",
                                    displayUpdateGranularity * 1000.0, 1.0 / minRefreshInterval);
                        isVRR = false;
                    }
                }
            }

            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Min/max refresh interval: %.2f/%.2f ms",
                        minRefreshInterval * 1000.0, maxRefreshInterval * 1000.0);
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Display update granularity: %.2f ms",
                        displayUpdateGranularity * 1000.0);
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Supports VRR: %s",
                        isVRR ? "yes" : (isProMotion ? "no, display is ProMotion" : "no, only one refresh interval"));
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Is fullscreen: %s",
                        isFullScreen ? "yes" : "no");

            m_SupportsVRR.store(false);

            if (isVRR) {
                if (isFullScreen) {
                    m_SupportsVRR.store(true);
                    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Output display: %s / VRR supported, refresh range %.2f-%.2f Hz",
                        [screen.localizedName UTF8String],
                        1.0 / maxRefreshInterval, 1.0 / minRefreshInterval);
                }
                else {
                    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Output display: %s @ %.2f Hz / VRR supported but inactive (not fullscreen)",
                        [screen.localizedName UTF8String], 1.0 / minRefreshInterval);
                }
            }
            else {
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Output display: %s @ %.2f Hz fixed",
                    [screen.localizedName UTF8String], 1.0 / minRefreshInterval);
            }

            setCurrentEDR(screen);

        #ifndef IMGUI_DISABLE
            DevUISettings::instance().SetConfig([=](DevUIConfig& config) {
                config.isVRR = isVRR;
                config.isFullscreen = isFullScreen;
                config.isProMotion = isProMotion;
            });
        #endif
        }
    }

    bool notifyWindowChanged(PWINDOW_STATE_CHANGE_INFO info) override
    {
        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "Metal renderer is handling window change: %dx%d on display %d",
                    info->width, info->height, info->displayIndex);

        auto unhandledStateFlags = info->stateChangeFlags;

        if (unhandledStateFlags & WINDOW_STATE_CHANGE_DISPLAY) {
            id<MTLDevice> displayDevice = copyWindowDisplayMetalDevice();
            if (displayDevice) {
                uint64_t newRegistryId = displayDevice.registryID;
                uint64_t currentRegistryId = m_MetalLayer.device.registryID;
                [displayDevice release];

                // Command queues, texture caches, pipelines, and textures are
                // all device-bound. Recreate the renderer instead of swapping
                // CAMetalLayer.device in place when the driving GPU changes.
                if (newRegistryId != currentRegistryId) {
                    SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                                "Forcing Metal renderer recreation after display GPU change");
                    return false;
                }
            }
        }

        refreshWindowMetadata();

        // We can always handle size changes
        unhandledStateFlags &= ~WINDOW_STATE_CHANGE_SIZE;

        // We can handle monitor changes
        unhandledStateFlags &= ~WINDOW_STATE_CHANGE_DISPLAY;

        // If nothing is left, we handled everything
        return unhandledStateFlags == 0;
    }

    NSWindow* getNSWindow()
    {
        NSWindow *nswindow = nil;
        SDL_SysWMinfo info;
        SDL_VERSION(&info.version);
        if (SDL_GetWindowWMInfo(m_Window, &info) && info.subsystem == SDL_SYSWM_COCOA) {
            nswindow = (__bridge NSWindow *)info.info.cocoa.window;
        }
        return nswindow;
    }

    void setPaused(bool isPaused)
    {
        m_IsPaused.store(isPaused);

        if (isPaused) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Metal renderer paused, window not visible");
        }
        else {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Metal renderer resumed");
        }
    }

    void setCurrentEDR(NSScreen* screen)
    {
        float maxPotentialEDR = screen.maximumPotentialExtendedDynamicRangeColorComponentValue;
        float maxReferenceEDR = screen.maximumReferenceExtendedDynamicRangeColorComponentValue;
        float referenceWhite = maxReferenceEDR > 1.0 ? 100.0f : 203.0f;
        m_MaxPotentialEDR.store(maxPotentialEDR);
        m_MaxReferenceEDR.store(maxReferenceEDR);
        m_ReferenceWhite.store(referenceWhite);

        float currentEDR = screen.maximumExtendedDynamicRangeColorComponentValue;
        if (currentEDR != m_CurrentEDR.load()) {
            m_CurrentEDR.store(currentEDR);
        }

        DevUISettings::instance().SetConfig([=](DevUIConfig& config) {
            config.isReferenceModeDisplay = maxReferenceEDR > 1.0;
            config.referenceWhite = referenceWhite;
        });
    }

    void startGPUCapture()
    {
        if (m_IsGPUCapturing) return;

        MTLCaptureManager *captureMgr = [MTLCaptureManager sharedCaptureManager];

        // capture all work from our command queue, but ignore ImGui
        MTLCaptureDescriptor *captureDesc = [[MTLCaptureDescriptor new] autorelease];
		captureDesc.captureObject = m_CommandQueue;

        const char* filePath = "/tmp/moonlight.gputrace";
		if (strlen(filePath)) {
			if ([captureMgr respondsToSelector: @selector(supportsDestination:)] &&
				[captureMgr supportsDestination: MTLCaptureDestinationGPUTraceDocument] ) {

				NSString* expandedFilePath = [[NSString stringWithUTF8String: filePath] stringByExpandingTildeInPath];
				SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Capturing GPU trace to file %s.", expandedFilePath.UTF8String);

				captureDesc.destination = MTLCaptureDestinationGPUTraceDocument;
				captureDesc.outputURL = [NSURL fileURLWithPath: expandedFilePath];

                NSError* error = nil;
                if (![captureMgr startCaptureWithDescriptor:captureDesc error:&error]) {
                    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                                 "Failed to start GPU capture: %s", error.localizedDescription.UTF8String);
                    return;
                }

                m_IsGPUCapturing = true;
			}
        }
    }

    void stopGPUCapture()
    {
        if (m_IsGPUCapturing) {
		    [[MTLCaptureManager sharedCaptureManager] stopCapture];
		    m_IsGPUCapturing = false;
        }
	}

    void releaseDrawable()
    {
        if (m_Drawable) {
            [m_Drawable release];
            m_Drawable = nil;
        }
    }

    void discardPendingFrame()
    {
        m_FrameEncoded = false;
        m_RenderPassDescriptor.colorAttachments[0].texture = nil;

        if (m_CommandBuffer[m_CurrentBuffer]) {
            [m_CommandBuffer[m_CurrentBuffer] release];
            m_CommandBuffer[m_CurrentBuffer] = nil;
        }
        releaseDrawable();
    }

    id<CAMetalDrawable> nextDrawable(bool forceNew=false)
    {
        if (forceNew) {
            // release the cached drawable so a new one will be requested
            releaseDrawable();
        }

        if (m_Drawable == nil) {
            id<CAMetalDrawable> drawable = [m_MetalLayer nextDrawable];
            if (!drawable) {
                return nil;
            }

            m_Drawable = [drawable retain];
            m_CurrentDrawableID = m_Drawable.drawableID;
        }

        return m_Drawable;
    }

    id<MTLCommandBuffer> getCommandBuffer()
    {
        if (!m_CommandBuffer[m_CurrentBuffer]) {
            m_CommandBuffer[m_CurrentBuffer] = [[m_CommandQueue commandBuffer] retain];
            m_CommandBuffer[m_CurrentBuffer].label = @"vt_metal command buffer";
        }
        return m_CommandBuffer[m_CurrentBuffer];
    }

    void applyDevUIConfig()
    {
    #ifndef IMGUI_DISABLE
        // we may need to apply changes from DevUI, but only check once per second
        uint64_t now = QpcNow();
        static uint64_t lastDevUI = 0;
        if (lastDevUI > 0 && QpcToMs(now - lastDevUI) < 1000.0) {
            return;
        }
        lastDevUI = now;

        auto cfg = DevUISettings::instance().GetConfig();
        auto markHdrMetadataChanged = [this]() {
            std::lock_guard<std::mutex> lock(m_HdrMetadataMutex);
            m_HdrMetadataChanged = true;
            ++m_HdrMetadataGeneration;
        };
        if (m_MaxFramesInFlight.load() != cfg.maxFramesInFlight) {
            int maxFramesInFlight = std::clamp(cfg.maxFramesInFlight, 2, 3);
            CAMetalLayer* layer = [m_MetalLayer retain];
            if (!runOnMainThreadCancellable(^{
                layer.maximumDrawableCount = maxFramesInFlight;
                [layer release];
            })) {
                return;
            }

            m_MaxFramesInFlight.store(maxFramesInFlight);
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Applied change of maxFramesInFlight to %d",
                        maxFramesInFlight);
        }

        if (m_ProMotionAllowsVRR.load() != cfg.proMotionAllowsVRR) {
            m_ProMotionAllowsVRR.store(cfg.proMotionAllowsVRR);
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Applied change of proMotionAllowsVRR to %d",
                        cfg.proMotionAllowsVRR);

            // Window/screen metadata and notification observers are AppKit
            // state. Recreate on the main thread instead of touching them from
            // this render-thread configuration callback.
            SDL_Event event;
            event.type = SDL_RENDER_DEVICE_RESET;
            SDL_PushEvent(&event);
            return;
        }

        if (m_UsePTSForVRR != cfg.usePTSForVRR) {
            m_UsePTSForVRR = cfg.usePTSForVRR;
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Applied change of usePTSForVRR to %d", m_UsePTSForVRR);
        }

        if (m_RequestedPresentMode != cfg.presentMode) {
            m_RequestedPresentMode = cfg.presentMode;
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Applied change of presentMode to %d", m_RequestedPresentMode);

            // change vsync if necessary
            bool vsyncEnabled = YES;
            if (m_RequestedPresentMode == StreamingPreferences::PRESENT_NO_VSYNC) {
                vsyncEnabled = NO;
            }
            if (m_IsVsync.load() != vsyncEnabled) {
                CAMetalLayer* layer = [m_MetalLayer retain];
                if (!runOnMainThreadCancellable(^{
                    layer.displaySyncEnabled = vsyncEnabled;
                    [layer release];
                })) {
                    return;
                }

                m_IsVsync.store(vsyncEnabled);
                SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                            "Applied change of vsync to %d",
                            vsyncEnabled);
            }
        }

        if (m_UseEDR != cfg.useEDR) {
            m_UseEDR = cfg.useEDR;
            markHdrMetadataChanged();

            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Applied change of useEDR to %d", m_UseEDR);
        }

        if (m_ReferenceWhite.load() != cfg.referenceWhite) {
            m_ReferenceWhite.store(cfg.referenceWhite);
            markHdrMetadataChanged();
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Applied change of ReferenceWhite to %.2f",
                        cfg.referenceWhite);
        }

        bool minNitsChanged = false;
        {
            std::lock_guard<std::mutex> lock(m_HdrMetadataMutex);
            if (m_MinNits != cfg.minNits) {
                m_MinNits = cfg.minNits;
                m_HdrMetadataChanged = true;
                m_OverrideNits = true;
                ++m_HdrMetadataGeneration;
                minNitsChanged = true;
            }
        }
        if (minNitsChanged) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Applied change of MinNits to %.4f",
                        cfg.minNits);
        }

        bool maxNitsChanged = false;
        {
            std::lock_guard<std::mutex> lock(m_HdrMetadataMutex);
            if (m_MaxNits != cfg.maxNits) {
                m_MaxNits = cfg.maxNits;
                m_HdrMetadataChanged = true;
                m_OverrideNits = true;
                ++m_HdrMetadataGeneration;
                maxNitsChanged = true;
            }
        }
        if (maxNitsChanged) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                        "Applied change of MaxNits to %.2f",
                        cfg.maxNits);
        }

        if (const char* env_capture = std::getenv("MTL_CAPTURE_ENABLED");
            env_capture && std::strcmp(env_capture, "1") == 0)
        {
            static bool captureGPUTrace = cfg.captureGPUTrace;
            if (captureGPUTrace != cfg.captureGPUTrace) {
                captureGPUTrace = cfg.captureGPUTrace;
                if (!m_IsGPUCapturing) {
                    startGPUCapture();
                }
                else {
                    stopGPUCapture();
                }
            }
        }

        if (m_ShowMetalHUD != cfg.showMetalHud) {
            SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Applied change of showMetalHud to %d", cfg.showMetalHud);
            m_ShowMetalHUD = cfg.showMetalHud;

            showMetalHud(m_ShowMetalHUD);
        }
    #endif
    }

private:
    bool m_HwAccel;
    SDL_Window* m_Window;
    AVBufferRef* m_HwContext;
    CAMetalLayer* m_MetalLayer;
    CVMetalTextureCacheRef m_TextureCache;
    CVMetalTextureRef m_CVMetalTextures[MAX_FRAMES_IN_FLIGHT][MAX_VIDEO_PLANES];
    id<MTLBuffer> m_CscParamsBuffer;
    id<MTLBuffer> m_VideoVertexBuffer;
    id<MTLTexture> m_OverlayTextures[Overlay::OverlayMax];
    SDL_SpinLock m_OverlayLock;
    MTLRenderPassDescriptor* m_RenderPassDescriptor;
    id<MTLRenderPipelineState> m_VideoPipelineState;
    id<MTLRenderPipelineState> m_OverlayPipelineState;
    id<MTLLibrary> m_ShaderLibrary;
    id<MTLCommandQueue> m_CommandQueue;
    id<MTLCommandBuffer> m_CommandBuffer[MAX_FRAMES_IN_FLIGHT];
    uint32_t m_CurrentBuffer;
    id<MTLTexture> m_SwMappingTextures[MAX_FRAMES_IN_FLIGHT][MAX_VIDEO_PLANES];
    SDL_MetalView m_MetalView;
    int m_LastFrameWidth;
    int m_LastFrameHeight;
    int m_LastDrawableWidth;
    int m_LastDrawableHeight;
    float m_RenderMaxNits;
    bool m_FrameEncoded;
    int64_t m_LastSubmittedFramePts;

    std::shared_ptr<PresentCallbackState> m_PresentState;
    std::atomic<int> m_MaxFramesInFlight;
    std::atomic<bool> m_IsFullScreen;
    std::atomic<bool> m_ProMotionAllowsVRR;
    bool m_UsePTSForVRR;
    bool m_UseEDR;
    bool m_ShowMetalHUD;
    std::atomic<float> m_MaxPotentialEDR;
    std::atomic<float> m_CurrentEDR;
    std::atomic<float> m_MaxReferenceEDR;
    std::atomic<float> m_ReferenceWhite;
    int m_RequestedPresentMode;
    std::atomic<CFTimeInterval> m_MinRefreshInterval;
    std::atomic<CFTimeInterval> m_MaxRefreshInterval;
    std::atomic<bool> m_SupportsVRR;
    std::atomic<bool> m_IsPaused;
    std::atomic<bool> m_IsVsync;
    std::atomic<double> m_VsyncTimestamp;
    std::atomic<double> m_VsyncDeadline;
    VTMetalObserver* m_Observer;
    bool m_IsGPUCapturing;
    id<CAMetalDrawable> m_Drawable;
    unsigned long m_CurrentDrawableID;
};

IFFmpegRenderer* VTMetalRendererFactory::createRenderer(bool hwAccel) {
    return new VTMetalRenderer(hwAccel);
}

@implementation VTMetalObserver {
    VTMetalRenderer* _renderer;
    id _note;
    id _note2;
}

- (id)initWithRenderer:(VTMetalRenderer *)renderer forWindow:(NSWindow *)window {
    self = [super init];
    if (self) {
        _renderer = renderer;

        void (^pauseIfHidden)(NSNotification*) = ^(NSNotification *note) {
            NSWindow* _window = (NSWindow *)note.object;
            bool shouldRender =
                _window.isVisible &&
                !_window.isMiniaturized &&
                (_window.occlusionState & NSWindowOcclusionStateVisible) != 0;
            VTMetalRenderer* renderer = _renderer;
            if (renderer) {
                renderer->setPaused(!shouldRender);
            }
        };

        void (^updateEDR)(NSNotification*) = ^(NSNotification*) {
            VTMetalRenderer* renderer = _renderer;
            NSScreen* screen = window.screen ?: NSScreen.mainScreen;
            if (renderer && screen) {
                renderer->setCurrentEDR(screen);
            }
        };

        // Pause rendering when off screen or hidden
        _note = [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowDidChangeOcclusionStateNotification
                                                                  object:window
                                                                   queue:nil
                                                              usingBlock:pauseIfHidden];
        // Watch for EDR changes
        _note2 = [[NSNotificationCenter defaultCenter] addObserverForName:NSApplicationDidChangeScreenParametersNotification
                                                                   object:NSApp
                                                                    queue:nil
                                                               usingBlock:updateEDR];
    }
    return self;
}

- (void)stop {
    _renderer = nullptr;

    if (_note) {
        [[NSNotificationCenter defaultCenter] removeObserver:_note];
        _note = nil;
    }
    if (_note2) {
        [[NSNotificationCenter defaultCenter] removeObserver:_note2];
        _note2 = nil;
    }
}

- (void)dealloc {
    [self stop];
    [super dealloc];
}

@end
