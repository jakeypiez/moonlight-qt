// Avoid conflict between AVFoundation and
// libavutil both defining AVMediaType
#define AVMediaType AVMediaType_FFmpeg
#include "displaylink_source.h"
#undef AVMediaType

#include <SDL_syswm.h>

#import <Availability.h>
#import <Cocoa/Cocoa.h>
#import <CoreVideo/CoreVideo.h>
#import <QuartzCore/CADisplayLink.h>

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
#define HAVE_MACOS_WINDOW_DISPLAY_LINK 1
#endif

@interface DisplayLinkTarget : NSObject
{
    DisplayLinkSource* _source;
#if HAVE_MACOS_WINDOW_DISPLAY_LINK
    CADisplayLink* _displayLink API_AVAILABLE(macos(14.0));
#endif
    CVDisplayLinkRef _legacyDisplayLink;
    double _legacyFallbackInterval;
    std::mutex _sourceMutex;
}

- (instancetype)initWithSource:(DisplayLinkSource*)source
                     forWindow:(NSWindow*)nswindow
                           fps:(double)fps;
#if HAVE_MACOS_WINDOW_DISPLAY_LINK
- (void)link:(CADisplayLink *)update API_AVAILABLE(macos(14.0));
#endif
- (void)legacyLinkOutputTime:(const CVTimeStamp *)outputTime;
- (void)stop;

@end

DisplayLinkSource::DisplayLinkSource()
    : m_DisplayLinkTarget(nullptr),
      m_TargetTimestamp(0.0)
{
}

DisplayLinkSource::~DisplayLinkSource()
{
    stop();
}

bool DisplayLinkSource::initialize(SDL_Window* window, double fps)
{
    stop();

    SDL_SysWMinfo info;
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(window, &info)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "DisplayLinkSource: SDL_GetWindowWMInfo() failed: %s",
                     SDL_GetError());
        return false;
    }

    NSWindow* nswindow = (__bridge NSWindow *)info.info.cocoa.window;
    if (!nswindow) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "DisplayLinkSource: Cocoa window is null");
        return false;
    }

    DisplayLinkTarget* target =
        [[DisplayLinkTarget alloc] initWithSource:this
                                        forWindow:nswindow
                                              fps:fps];
    if (!target) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                     "DisplayLinkSource: error creating DisplayLink");
        return false;
    }

    m_DisplayLinkTarget = target;
    m_TargetTimestamp.store(0.0);

    return true;
}

void DisplayLinkSource::stop()
{
    m_TargetTimestamp.store(0.0);

    DisplayLinkTarget* target = (DisplayLinkTarget*)m_DisplayLinkTarget;
    if (target) {
        [target stop];
        [target release];
        m_DisplayLinkTarget = nullptr;
    }
}

bool DisplayLinkSource::isAsync()
{
    return true;
}

void DisplayLinkSource::displayLinkUpdate(double timestamp, double targetTimestamp)
{
    if (targetTimestamp <= timestamp) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mtx);
    m_TargetTimestamp.store(targetTimestamp);

    FramePacer::instance().signalVsyncTS(timestamp, targetTimestamp);
}

///////

static CVReturn legacyDisplayLinkCallback(CVDisplayLinkRef,
                                          const CVTimeStamp*,
                                          const CVTimeStamp* outputTime,
                                          CVOptionFlags,
                                          CVOptionFlags*,
                                          void* context)
{
    @autoreleasepool {
        DisplayLinkTarget* target = static_cast<DisplayLinkTarget*>(context);
        [target legacyLinkOutputTime:outputTime];
    }
    return kCVReturnSuccess;
}

@implementation DisplayLinkTarget

- (instancetype)initWithSource:(DisplayLinkSource*)source
                     forWindow:(NSWindow*)window
                            fps:(double)fps
{
    self = [super init];
    if (self) {
        _source = source;
        _legacyDisplayLink = nullptr;
        _legacyFallbackInterval = fps > 0.0 ? 1.0 / fps : 1.0 / 60.0;

#if HAVE_MACOS_WINDOW_DISPLAY_LINK
        if (@available(macOS 14.0, *)) {
            _displayLink = [window displayLinkWithTarget:self
                                                selector:@selector(link:)];
            if (_displayLink) {
                _displayLink.preferredFrameRateRange = CAFrameRateRangeMake(fps, fps, fps);
                [_displayLink addToRunLoop:[NSRunLoop mainRunLoop]
                                   forMode:NSRunLoopCommonModes];
                return self;
            }
        }
#endif

        // NSWindow display links are only available on macOS 14+. Preserve
        // compatibility with the app's macOS 11 deployment target by using
        // the precise CoreVideo host clock on earlier systems.
        NSScreen* screen = window.screen ?: NSScreen.mainScreen;
        NSNumber* screenNumber = screen.deviceDescription[@"NSScreenNumber"];
        if (!screenNumber) {
            [self release];
            return nil;
        }

        CGDirectDisplayID displayId = static_cast<CGDirectDisplayID>(screenNumber.unsignedIntValue);
        CVReturn status = CVDisplayLinkCreateWithCGDisplay(displayId, &_legacyDisplayLink);
        if (status == kCVReturnSuccess) {
            status = CVDisplayLinkSetOutputCallback(_legacyDisplayLink,
                                                    legacyDisplayLinkCallback,
                                                    self);
        }
        if (status == kCVReturnSuccess) {
            status = CVDisplayLinkStart(_legacyDisplayLink);
        }
        if (status != kCVReturnSuccess) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,
                         "DisplayLinkSource: CVDisplayLink setup failed: %d",
                         status);
            [self release];
            return nil;
        }

        SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION,
                    "DisplayLinkSource: using CoreVideo compatibility path");
    }
    return self;
}

#if HAVE_MACOS_WINDOW_DISPLAY_LINK
- (void)link:(CADisplayLink*)update
{
    std::lock_guard<std::mutex> lock(_sourceMutex);
    if (_source) {
        _source->displayLinkUpdate((double)update.timestamp,
                                   (double)update.targetTimestamp);
    }
}
#endif

- (void)legacyLinkOutputTime:(const CVTimeStamp*)outputTime
{
    std::lock_guard<std::mutex> lock(_sourceMutex);
    if (!_source ||
            !(outputTime->flags & kCVTimeStampHostTimeValid)) {
        return;
    }

    const double frequency = CVGetHostClockFrequency();
    if (frequency <= 0.0) {
        return;
    }

    // CVDisplayLink's `now` timestamp describes when the callback runs, not
    // the preceding vblank. Derive exactly one refresh period from CoreVideo's
    // display timing instead. Callback-to-callback time can span multiple
    // refreshes when a callback is delayed, so it is unsuitable here.
    double interval = 0.0;
    if ((outputTime->flags & kCVTimeStampVideoRefreshPeriodValid) &&
            outputTime->videoTimeScale > 0 && outputTime->videoRefreshPeriod > 0) {
        interval = (double)outputTime->videoRefreshPeriod /
                   (double)outputTime->videoTimeScale;
        if ((outputTime->flags & kCVTimeStampRateScalarValid) &&
                outputTime->rateScalar > 0.0) {
            interval /= outputTime->rateScalar;
        }
    }

    if (interval <= 0.0 || interval >= 1.0) {
        interval = CVDisplayLinkGetActualOutputVideoRefreshPeriod(_legacyDisplayLink);
    }
    if (interval <= 0.0 || interval >= 1.0) {
        CVTime nominal = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(_legacyDisplayLink);
        if (!(nominal.flags & kCVTimeIsIndefinite) &&
                nominal.timeValue > 0 && nominal.timeScale > 0) {
            interval = (double)nominal.timeValue / (double)nominal.timeScale;
        }
    }
    if (interval <= 0.0 || interval >= 1.0) {
        interval = _legacyFallbackInterval;
    }

    double deadline = (double)outputTime->hostTime / frequency;
    _source->displayLinkUpdate(deadline - interval, deadline);
}

- (void)stop
{
#if HAVE_MACOS_WINDOW_DISPLAY_LINK
    if (_displayLink) {
        [_displayLink invalidate];
        _displayLink = nil;
    }
#endif

    if (_legacyDisplayLink) {
        CVDisplayLinkStop(_legacyDisplayLink);
    }

    // Wait for an in-progress callback before invalidating the renderer
    // pointer. This prevents a callback that already loaded _source from
    // racing renderer teardown.
    {
        std::lock_guard<std::mutex> lock(_sourceMutex);
        _source = nullptr;
    }

    if (_legacyDisplayLink) {
        CVDisplayLinkSetOutputCallback(_legacyDisplayLink, nullptr, nullptr);
        CFRelease(_legacyDisplayLink);
        _legacyDisplayLink = nullptr;
    }
}

- (void)dealloc
{
    [self stop];
    [super dealloc];
}

@end
