/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Contains the main app implementation using Vision.
*/

import UIKit
import AVKit
import Vision

class ViewController: UIViewController {
    
    // Main view for showing camera content.
    @IBOutlet weak var previewView: UIView?
    
    // AVCapture variables to hold sequence data
    var session: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    var videoDataOutput: AVCaptureVideoDataOutput?
    var videoDataOutputQueue: DispatchQueue?
    
    var captureDevice: AVCaptureDevice?
    var captureDeviceResolution: CGSize = CGSize()
    
    // Layer UI for drawing Vision results
    var rootLayer: CALayer?
    var detectionOverlayLayer: CALayer?
    var detectedFaceRectangleShapeLayer: CAShapeLayer?
    var detectedFaceLandmarksShapeLayer: CAShapeLayer?
    
    // Vision requests
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?
    private var trackingRequests: [VNTrackObjectRequest]?
    
    // Tracks if the mouth is open in a way that suggests a surprised expression
    var isMouthOpenForSurprise = false
    
    // Indicates whether the eyes are currently blinking
    var isBlinking = false
    
    // Detects if the head is moving side-to-side, indicating a "shaking head" gesture
    var isShakingHead = false
    
    // Records the last time a facial feature was detected, to track timing and reduce redundant detections
    var lastDetectedTime: Date?
    
    
    
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()
    
    // MARK: UIViewController overrides
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup video for high resolution, drop frames when busy, and front camera
        self.session = self.setupAVCaptureSession()
        
        // setup the vision objects for (1) detection and (2) tracking
        self.prepareVisionRequest()
        
        // start the capture session and get processing a face!
        self.session?.startRunning()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .portrait
    }
    
    
    // MARK: Performing Vision Requests
    
    /// - Tag: WriteCompletionHandler
    fileprivate func prepareVisionRequest() {
        
        self.trackingRequests = []
        
        // create a detection request that processes an image and returns face features
        // completion handler does not run immediately, it is run
        // after a face is detected
        let faceDetectionRequest:VNDetectFaceRectanglesRequest = VNDetectFaceRectanglesRequest(completionHandler: self.faceDetectionCompletionHandler)
        
        // Save this detection request for later processing
        self.detectionRequests = [faceDetectionRequest]
        
        // setup the tracking of a sequence of features from detection
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        
        // setup drawing layers for showing output of face detection
        self.setupVisionDrawingLayers()
    }
    
    // define behavior for when we detect a face
    func faceDetectionCompletionHandler(request:VNRequest, error: Error?){
        // any errors? If yes, show and try to keep going
        if error != nil {
            print("FaceDetection error: \(String(describing: error)).")
        }
        
        // see if we can get any face features, this will fail if no faces detected
        // try to save the face observations to a results vector
        guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
              let results = faceDetectionRequest.results as? [VNFaceObservation] else {
            return
        }
        
        if !results.isEmpty{
            print("Initial Face found... setting up tracking.")
            
            
        }
        
        // if we got here, then a face was detected and we have its features saved
        // The above face detection was the most computational part of what we did
        // the remaining tracking only needs the results vector of face features
        // so we can process it in the main queue (because we will us it to update UI)
        DispatchQueue.main.async {
            // Add the face features to the tracking list
            for observation in results {
                let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                // the array starts empty, but this will constantly add to it
                // since on the main queue, there are no race conditions
                // everything is from a single thread
                // once we add this, it kicks off tracking in another function
                self.trackingRequests?.append(faceTrackingRequest)
                
                // NOTE: if the initial face detection is actually not a face,
                // then the app will continually mess up trying to perform tracking
            }
        }
        
    }
    
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    /// - Tag: PerformRequests
    // Handle delegate method callback on receiving a sample buffer.
    // This is where we get the pixel buffer from the camera and need to
    // generate the vision requests
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection)
    {
        
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]
        
        // see if camera has any instrinsic transforms on it
        // if it does, add these to the options for requests
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        // check to see if we can get the pixels for processing, else return
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        // get portrait orientation for UI
        let exifOrientation = self.exifOrientationForCurrentDeviceOrientation()
        
        guard let requests = self.trackingRequests else {
            print("Tracking request array not setup, aborting.")
            return
        }
        
        
        // check to see if the tracking request is empty (no face currently detected)
        // if it is empty,
        if requests.isEmpty{
            // No tracking object detected, so perform initial detection
            // the initial detection takes some time to perform
            // so we special case it here
            
            self.performInitialDetection(pixelBuffer: pixelBuffer,
                                         exifOrientation: exifOrientation,
                                         requestHandlerOptions: requestHandlerOptions)
            
            return  // just perform the initial request
        }
        
        // if tracking was not empty, it means we have detected a face very recently
        // so no we can process the sequence of tracking face features
        
        self.performTracking(requests: requests,
                             pixelBuffer: pixelBuffer,
                             exifOrientation: exifOrientation)
        
        
        // if there are no valid observations, then this will be empty
        // the function above will empty out all the elements
        // in our tracking if nothing is high confidence in the output
        if let newTrackingRequests = self.trackingRequests {
            
            if newTrackingRequests.isEmpty {
                // Nothing was high enough confidence to track, just abort.
                print("Face object lost, resetting detection...")
                return
            }
            
            self.performLandmarkDetection(newTrackingRequests: newTrackingRequests,
                                          pixelBuffer: pixelBuffer,
                                          exifOrientation: exifOrientation,
                                          requestHandlerOptions: requestHandlerOptions)
            
        }
        
        
    }
    
    // functionality to run the image detection on pixel buffer
    // This is an involved computation, so beware of running too often
    func performInitialDetection(pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {
        // create request
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                        orientation: exifOrientation,
                                                        options: requestHandlerOptions)
        
        do {
            if let detectRequests = self.detectionRequests{
                // try to detect face and add it to tracking buffer
                try imageRequestHandler.perform(detectRequests)
            }
        } catch let error as NSError {
            NSLog("Failed to perform FaceRectangleRequest: %@", error)
        }
    }
    
    
    // this function performs all the tracking of the face sequence
    func performTracking(requests:[VNTrackObjectRequest],
                         pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation)
    {
        do {
            // perform tracking on the pixel buffer, which is
            // less computational than fully detecting a face
            // if a face was not correct initially, this tracking
            //   will also be not great... but it is fast!
            try self.sequenceRequestHandler.perform(requests,
                                                    on: pixelBuffer,
                                                    orientation: exifOrientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
        
        // if there are any tracking results, let's process them here
        
        // Setup the next round of tracking.
        var newTrackingRequests = [VNTrackObjectRequest]()
        for trackingRequest in requests {
            
            // any valid results in the request?
            // if so, grab the first request
            if let results = trackingRequest.results,
               let observation = results[0] as? VNDetectedObjectObservation {
                
                
                // is this tracking request of high confidence?
                // If it is, then we should add it to processing buffer
                // the threshold is arbitrary. You can adjust to you liking
                if !trackingRequest.isLastFrame {
                    if observation.confidence > 0.3 {
                        trackingRequest.inputObservation = observation
                    }
                    else {
                        
                        // once below thresh, make it last frame
                        // this will stop the processing of tracker
                        trackingRequest.isLastFrame = true
                    }
                    // add to running tally of high confidence observations
                    newTrackingRequests.append(trackingRequest)
                }
                
            }
            
        }
        self.trackingRequests = newTrackingRequests
        
        
    }
    
    func performLandmarkDetection(newTrackingRequests:[VNTrackObjectRequest], pixelBuffer:CVPixelBuffer, exifOrientation:CGImagePropertyOrientation, requestHandlerOptions:[VNImageOption: AnyObject]) {
        // Perform face landmark tracking on detected faces.
        // setup an empty arry for now
        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()
        
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            
            // create a request for facial landmarks
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: self.landmarksCompletionHandler)
            
            // get tracking result and observation for result
            if let trackingResults = trackingRequest.results,
               let observation = trackingResults[0] as? VNDetectedObjectObservation{
                
                // save the observation info
                let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)
                
                // set information for face
                faceLandmarksRequest.inputFaceObservations = [faceObservation]
                
                // Continue to track detected facial landmarks.
                faceLandmarkRequests.append(faceLandmarksRequest)
                
                // setup for performing landmark detection
                let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                                orientation: exifOrientation,
                                                                options: requestHandlerOptions)
                
                do {
                    // try to find landmarks in face, then display in completion handler
                    try imageRequestHandler.perform(faceLandmarkRequests)
                    
                    // completion handler will now take over and finish the job!
                } catch let error as NSError {
                    NSLog("Failed to perform FaceLandmarkRequest: %@", error)
                }
            }
        }
    }
    
    
    // MARK: - Landmark Detection Completion Handler
    func landmarksCompletionHandler(request: VNRequest, error: Error?) {
        if let error = error {
            print("FaceLandmarks error: \(String(describing: error)).")
            return
        }
        
        guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
              let results = landmarksRequest.results as? [VNFaceObservation] else { return }
        
        DispatchQueue.main.async {
            self.drawFaceObservations(results)
            for faceObservation in results {
                if let landmarks = faceObservation.landmarks {
                    self.processLandmarks(landmarks)
                }
            }
        }
    }
    
    // MARK: - Process Detected Landmarks
    func processLandmarks(_ landmarks: VNFaceLandmarks2D) {
        guard let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let outerLips = landmarks.outerLips else { return }
        
        if let lastTime = lastDetectedTime, Date().timeIntervalSince(lastTime) < 0.3 {
            return
        }
        lastDetectedTime = Date()
        
        // Apply refined expression detection and update UI accordingly
        updateUIIfChanged(condition: isMouthOpenForSurprise(outerLips), currentState: &isMouthOpenForSurprise, action: displayOpenMouthUI)
        updateUIIfChanged(condition: isBlinking(leftEye, rightEye), currentState: &isBlinking, action: displayBlinkUI)
        updateUIIfChanged(condition: isShakingHead(leftEye, rightEye), currentState: &isShakingHead, action: displayShakeHeadUI)
    }
    
    // MARK: - Refined Expression Detection Conditions
    
    func isMouthOpenForSurprise(_ outerLips: VNFaceLandmarkRegion2D) -> Bool {
        let points = outerLips.normalizedPoints
        guard points.count > 9 else { return false } // Ensure there are enough points
        
        // Define left, right, and center points for upper and lower lips
        let leftCorner = points[0]
        let rightCorner = points[6]
        let upperMiddle = points[3]
        let lowerMiddle = points[9]
        
        // Calculate the mouth width and height
        let mouthWidth = abs(rightCorner.x - leftCorner.x)
        let mouthHeight = abs(upperMiddle.y - lowerMiddle.y)
        
        // Heuristic for surprise: A significantly larger vertical distance relative to width
        return mouthHeight / mouthWidth > 0.5 && mouthHeight > 0.3
    }
    
    
    // Checks if both eyes are in a closed state, suggesting a blink
    func isBlinking(_ leftEye: VNFaceLandmarkRegion2D, _ rightEye: VNFaceLandmarkRegion2D) -> Bool {
        // Get the normalized points of each eye's landmarks
        let leftEyePoints = leftEye.normalizedPoints
        let rightEyePoints = rightEye.normalizedPoints
        // Ensure there are enough points to measure eye height
        guard leftEyePoints.count > 4, rightEyePoints.count > 4 else { return false }
        
        // Calculate the vertical distance between specific eye points to estimate eye openness
        let leftEyeHeight = abs(leftEyePoints[4].y - leftEyePoints[1].y)
        let rightEyeHeight = abs(rightEyePoints[4].y - rightEyePoints[1].y)
        
        // Return true if both eyes are closed enough to indicate a blink
        return leftEyeHeight < 0.02 && rightEyeHeight < 0.02 // Sensitivity adjusted for more accurate blink detection
    }
    
    // Detects if the head is moving side-to-side, indicating a "shaking head" gesture
    func isShakingHead(_ leftEye: VNFaceLandmarkRegion2D, _ rightEye: VNFaceLandmarkRegion2D) -> Bool {
        // Get the normalized points of each eye's landmarks
        let leftEyePoints = leftEye.normalizedPoints
        let rightEyePoints = rightEye.normalizedPoints
        // Ensure there are enough points to measure the distance between the eyes
        guard leftEyePoints.count > 0, rightEyePoints.count > 0 else { return false }
        
        // Calculate the horizontal distance between the two eyes
        let horizontalEyeDistance = abs(leftEyePoints[0].x - rightEyePoints[0].x)
        
        // Return true if distance exceeds threshold, suggesting head movement
        return horizontalEyeDistance > 0.35 // Threshold lowered to improve head shake detection sensitivity
    }
    
    // MARK: - UI Update Methods
    
    // Updates the UI when a condition changes and triggers the associated action
    private func updateUIIfChanged(condition: Bool, currentState: inout Bool, action: () -> Void) {
        // Check if the new condition differs from the current state
        if condition != currentState {
            currentState = condition
            // Execute the UI action if the condition is now true
            if condition { action() }
        }
    }
    
    // MARK: - UI Feedback Display Methods
    
    // Displays a UI overlay and feedback for open mouth detection
    func displayOpenMouthUI() {
        updateOverlayColor(to: .red)
        showUIFeedback(labelText: "😮 Detected!", color: .red, positionY: 100)
    }
    
    // Displays a UI overlay and feedback for blink detection
    func displayBlinkUI() {
        updateOverlayColor(to: .blue)
        showUIFeedback(labelText: "😉 Blink Detected!", color: .blue, positionY: 200)
    }
    
    // Displays a UI overlay and feedback for head shake detection
    func displayShakeHeadUI() {
        updateOverlayColor(to: .purple)
        showUIFeedback(labelText: "🙅‍♂️ Head Shake Detected!", color: .purple, positionY: 300)
    }
    
    // Updates the color of the overlay to visually represent detected gestures
    private func updateOverlayColor(to color: UIColor) {
        detectionOverlayLayer?.sublayers?.forEach { layer in
            if let shapeLayer = layer as? CAShapeLayer {
                shapeLayer.strokeColor = color.cgColor
            }
        }
    }
    
    // Displays animated feedback text on the UI with fade-in and fade-out animations
    private func showUIFeedback(labelText: String, color: UIColor, positionY: CGFloat) {
        // Create a label with the specified text, color, and position
        let label = UILabel()
        label.text = labelText
        label.font = UIFont.systemFont(ofSize: 24)
        label.textColor = color
        label.alpha = 0 // Start label as invisible
        label.frame = CGRect(x: 50, y: positionY, width: 250, height: 50)
        self.view.addSubview(label)
        
        // Animate the label's appearance and disappearance
        UIView.animate(withDuration: 0.5, animations: {
            label.alpha = 1 // Fade in
        }) { _ in
            UIView.animate(withDuration: 1.0, delay: 1.0, options: [], animations: {
                label.alpha = 0 // Fade out
            }) { _ in
                label.removeFromSuperview() // Remove the label after fading out
            }
        }
    }
}
    



// MARK: Helper Methods
extension UIViewController{
    
    // Helper Methods for Error Presentation
    
    fileprivate func presentErrorAlert(withTitle title: String = "Unexpected Failure", message: String) {
        let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
        self.present(alertController, animated: true)
    }
    
    fileprivate func presentError(_ error: NSError) {
        self.presentErrorAlert(withTitle: "Failed with error \(error.code)", message: error.localizedDescription)
    }
    
    // Helper Methods for Handling Device Orientation & EXIF
    
    fileprivate func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
    func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        
        switch deviceOrientation {
        case .portraitUpsideDown:
            return .rightMirrored
            
        case .landscapeLeft:
            return .downMirrored
            
        case .landscapeRight:
            return .upMirrored
            
        default:
            return .leftMirrored
        }
    }
    
    func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
        return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
    }
}


// MARK: Extension for AVCapture Setup
extension ViewController:AVCaptureVideoDataOutputSampleBufferDelegate{
    
    
    /// - Tag: CreateCaptureSession
    fileprivate func setupAVCaptureSession() -> AVCaptureSession? {
        let captureSession = AVCaptureSession()
        do {
            let inputDevice = try self.configureFrontCamera(for: captureSession)
            self.configureVideoDataOutput(for: inputDevice.device, resolution: inputDevice.resolution, captureSession: captureSession)
            self.designatePreviewLayer(for: captureSession)
            return captureSession
        } catch let executionError as NSError {
            self.presentError(executionError)
        } catch {
            self.presentErrorAlert(message: "An unexpected failure has occured")
        }
        
        self.teardownAVCapture()
        
        return nil
    }
    
    /// - Tag: ConfigureDeviceResolution
    fileprivate func highestResolution420Format(for device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, resolution: CGSize)? {
        var highestResolutionFormat: AVCaptureDevice.Format? = nil
        var highestResolutionDimensions = CMVideoDimensions(width: 0, height: 0)
        
        for format in device.formats {
            let deviceFormat = format as AVCaptureDevice.Format
            
            let deviceFormatDescription = deviceFormat.formatDescription
            if CMFormatDescriptionGetMediaSubType(deviceFormatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                let candidateDimensions = CMVideoFormatDescriptionGetDimensions(deviceFormatDescription)
                if (highestResolutionFormat == nil) || (candidateDimensions.width > highestResolutionDimensions.width) {
                    highestResolutionFormat = deviceFormat
                    highestResolutionDimensions = candidateDimensions
                }
            }
        }
        
        if highestResolutionFormat != nil {
            let resolution = CGSize(width: CGFloat(highestResolutionDimensions.width), height: CGFloat(highestResolutionDimensions.height))
            return (highestResolutionFormat!, resolution)
        }
        
        return nil
    }
    
    fileprivate func configureFrontCamera(for captureSession: AVCaptureSession) throws -> (device: AVCaptureDevice, resolution: CGSize) {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        
        if let device = deviceDiscoverySession.devices.first {
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                }
                
                if let highestResolution = self.highestResolution420Format(for: device) {
                    try device.lockForConfiguration()
                    device.activeFormat = highestResolution.format
                    device.unlockForConfiguration()
                    
                    return (device, highestResolution.resolution)
                }
            }
        }
        
        throw NSError(domain: "ViewController", code: 1, userInfo: nil)
    }
    
    /// - Tag: CreateSerialDispatchQueue
    fileprivate func configureVideoDataOutput(for inputDevice: AVCaptureDevice, resolution: CGSize, captureSession: AVCaptureSession) {
        
        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        // Create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured.
        // A serial dispatch queue must be used to guarantee that video frames will be delivered in order.
        let videoDataOutputQueue = DispatchQueue(label: "com.example.apple-samplecode.VisionFaceTrack")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        videoDataOutput.connection(with: .video)?.isEnabled = true
        
        if let captureConnection = videoDataOutput.connection(with: AVMediaType.video) {
            if captureConnection.isCameraIntrinsicMatrixDeliverySupported {
                captureConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        
        self.videoDataOutput = videoDataOutput
        self.videoDataOutputQueue = videoDataOutputQueue
        
        self.captureDevice = inputDevice
        self.captureDeviceResolution = resolution
    }
    
    /// - Tag: DesignatePreviewLayer
    fileprivate func designatePreviewLayer(for captureSession: AVCaptureSession) {
        let videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer = videoPreviewLayer
        
        videoPreviewLayer.name = "CameraPreview"
        videoPreviewLayer.backgroundColor = UIColor.black.cgColor
        videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        
        if let previewRootLayer = self.previewView?.layer {
            self.rootLayer = previewRootLayer
            
            previewRootLayer.masksToBounds = true
            videoPreviewLayer.frame = previewRootLayer.bounds
            previewRootLayer.addSublayer(videoPreviewLayer)
        }
    }
    
    // Removes infrastructure for AVCapture as part of cleanup.
    fileprivate func teardownAVCapture() {
        self.videoDataOutput = nil
        self.videoDataOutputQueue = nil
        
        if let previewLayer = self.previewLayer {
            previewLayer.removeFromSuperlayer()
            self.previewLayer = nil
        }
    }
}


// MARK: Extension Drawing Vision Observations
extension ViewController {
    
    
    fileprivate func setupVisionDrawingLayers() {
        let captureDeviceResolution = self.captureDeviceResolution
        
        let captureDeviceBounds = CGRect(x: 0,
                                         y: 0,
                                         width: captureDeviceResolution.width,
                                         height: captureDeviceResolution.height)
        
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX,
                                                     y: captureDeviceBounds.midY)
        
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        
        guard let rootLayer = self.rootLayer else {
            self.presentErrorAlert(message: "view was not property initialized")
            return
        }
        
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 0.7
        faceRectangleShapeLayer.shadowRadius = 5
        
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"
        faceLandmarksShapeLayer.bounds = captureDeviceBounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint
        faceLandmarksShapeLayer.fillColor = nil
        faceLandmarksShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor
        faceLandmarksShapeLayer.lineWidth = 3
        faceLandmarksShapeLayer.shadowOpacity = 0.7
        faceLandmarksShapeLayer.shadowRadius = 5
        
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)
        rootLayer.addSublayer(overlayLayer)
        
        self.detectionOverlayLayer = overlayLayer
        self.detectedFaceRectangleShapeLayer = faceRectangleShapeLayer
        self.detectedFaceLandmarksShapeLayer = faceLandmarksShapeLayer
        
        self.updateLayerGeometry()
    }
    
    fileprivate func updateLayerGeometry() {
        guard let overlayLayer = self.detectionOverlayLayer,
            let rootLayer = self.rootLayer,
            let previewLayer = self.previewLayer
            else {
            return
        }
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        
        // Rotate the layer into screen orientation.
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
            rotation = 180
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
            
        case .landscapeLeft:
            rotation = 90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
            
        case .landscapeRight:
            rotation = -90
            scaleX = videoPreviewRect.height / captureDeviceResolution.width
            scaleY = scaleX
            
        default:
            rotation = 0
            scaleX = videoPreviewRect.width / captureDeviceResolution.width
            scaleY = videoPreviewRect.height / captureDeviceResolution.height
        }
        
        // Scale and mirror the image to ensure upright presentation.
        let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation))
            .scaledBy(x: scaleX, y: -scaleY)
        overlayLayer.setAffineTransform(affineTransform)
        
        // Cover entire screen UI.
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
    }
    
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        let pointCount = landmarkRegion.pointCount
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints
            path.move(to: points[0], transform: affineTransform)
            path.addLines(between: points, transform: affineTransform)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath()
            }
        }
    }
    
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        let displaySize = self.captureDeviceResolution
        
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        faceRectanglePath.addRect(faceBounds)
        
        if let landmarks = faceObservation.landmarks {
            // Landmarks are relative to -- and normalized within --- face bounds
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Treat eyebrows and lines as open-ended regions when drawing paths.
            let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEyebrow,
                landmarks.rightEyebrow,
                landmarks.faceContour,
                landmarks.noseCrest,
                landmarks.medianLine
            ]
            for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
                self.addPoints(in: openLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: false)
            }
            
            // Draw eyes, lips, and nose as closed regions.
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEye,
                landmarks.rightEye,
                landmarks.outerLips,
                landmarks.innerLips,
                landmarks.nose
            ]
            for closedLandmarkRegion in closedLandmarkRegions where closedLandmarkRegion != nil {
                self.addPoints(in: closedLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: true)
            }
        }
    }
    
    /// - Tag: DrawPaths
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        guard let faceRectangleShapeLayer = self.detectedFaceRectangleShapeLayer,
            let faceLandmarksShapeLayer = self.detectedFaceLandmarksShapeLayer
            else {
            return
        }
        
        CATransaction.begin()
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        for faceObservation in faceObservations {
            self.addIndicators(to: faceRectanglePath,
                               faceLandmarksPath: faceLandmarksPath,
                               for: faceObservation)
        }
        
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        self.updateLayerGeometry()
        
        CATransaction.commit()
    }
}





    
