import * as faceapi from 'face-api.js';
import { pipeline, env } from '@huggingface/transformers';

// Configure Transformers.js
env.allowLocalModels = false;

// Configuration
const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
const STORAGE_KEY = 'registered_face_descriptor';
const MATCH_THRESHOLD = 0.6; // Lower = stricter matching
const VLM_CAPTURE_INTERVAL = 3000; // 3 seconds between frame captures

// State
let modelsLoaded = false;
let registerVideo = null;
let monitorVideo = null;
let isMonitoring = false;
let monitorInterval = null;

// VLM State
let vlmPipeline = null;
let vlmLoading = false;
let vlmEnabled = false;
let vlmCaptureInterval = null;

// Liveness State
let livenessVideo = null;
let livenessCanvas = null;
let isLivenessCheck = false;
let livenessInterval = null;
let currentChallenge = 0;
let challenges = [];
let challengeTimeout = null;
let blinkCount = 0;
let lastEAR = 0;
let eyesWereClosed = false;

// Model files to load
const modelFiles = [
  { name: 'Tiny Face Detector', files: ['tiny_face_detector_model-weights_manifest.json', 'tiny_face_detector_model-shard1'] },
  { name: 'Face Landmark 68', files: ['face_landmark_68_model-weights_manifest.json', 'face_landmark_68_model-shard1'] },
  { name: 'Face Recognition', files: ['face_recognition_model-weights_manifest.json', 'face_recognition_model-shard1', 'face_recognition_model-shard2'] }
];

// ============================================
// Model Loading with Progress
// ============================================

async function loadModelsWithProgress() {
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const progressDetail = document.getElementById('progress-detail');
  
  let totalFiles = modelFiles.reduce((acc, m) => acc + m.files.length, 0);
  let loadedFiles = 0;

  progressDetail.textContent = 'Starting download...';

  // Create custom fetch to track progress
  const originalFetch = window.fetch;
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    
    if (args[0].includes('weights') && args[0].includes('.json')) {
      const modelName = modelFiles.find(m => args[0].includes(m.files[0]))?.name || 'Model';
      progressDetail.textContent = `Loading ${modelName}...`;
    }
    
    return response;
  };

  try {
    // Load models sequentially with progress updates
    const totalModels = 3;
    
    progressDetail.textContent = 'Loading Tiny Face Detector...';
    progressFill.style.width = '10%';
    progressText.textContent = '10%';
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    
    progressDetail.textContent = 'Loading Face Landmarks...';
    progressFill.style.width = '40%';
    progressText.textContent = '40%';
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    
    progressDetail.textContent = 'Loading Face Recognition...';
    progressFill.style.width = '70%';
    progressText.textContent = '70%';
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    
    progressFill.style.width = '100%';
    progressText.textContent = '100%';
    progressDetail.textContent = 'Models loaded successfully!';
    
    // Restore original fetch
    window.fetch = originalFetch;
    
    modelsLoaded = true;
    
    // Short delay to show completion
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return true;
  } catch (error) {
    window.fetch = originalFetch;
    progressDetail.textContent = `Error: ${error.message}`;
    progressFill.style.background = 'var(--accent-danger)';
    throw error;
  }
}

// ============================================
// VLM (Vision Language Model) Functions
// ============================================

function updateAIStatus(message, type = 'loading') {
  const statusEl = document.getElementById('ai-status');
  statusEl.textContent = message;
  statusEl.className = `ai-status ${type}`;
  statusEl.classList.remove('hidden');
}

function hideAIStatus() {
  const statusEl = document.getElementById('ai-status');
  statusEl.classList.add('hidden');
}

function showAIDescription(text) {
  const container = document.getElementById('ai-description-container');
  const descriptionEl = document.getElementById('ai-description');
  
  container.classList.remove('hidden');
  descriptionEl.innerHTML = text;
}

async function loadVLM() {
  if (vlmPipeline || vlmLoading) return;
  
  vlmLoading = true;
  updateAIStatus('Initializing AI...', 'loading');
  
  try {
    // Skip WebGPU - too unreliable, use WASM directly
    updateAIStatus('Loading AI model (WASM)...', 'loading');
    
    // Add timeout wrapper
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Model load timeout (30s)')), 30000);
    });
    
    // Use WASM for reliability
    const loadPromise = pipeline(
      'image-to-text',
      'Xenova/vit-gpt2-image-captioning',
      {
        device: 'wasm',
        progress_callback: (progress) => {
          if (progress.status === 'downloading') {
            const percent = progress.progress ? Math.round(progress.progress) : 0;
            updateAIStatus(`Downloading model... ${percent}%`, 'loading');
          } else if (progress.status === 'loading') {
            updateAIStatus('Initializing model...', 'loading');
          }
        }
      }
    );
    
    updateAIStatus('Downloading AI model...', 'loading');
    vlmPipeline = await Promise.race([loadPromise, timeoutPromise]);
    
    updateAIStatus('Model ready!', 'loading');
    await new Promise(r => setTimeout(r, 500));
    
    hideAIStatus();
    showAIDescription('<span class="ai-placeholder">AI ready. Capturing scene...</span>');
    
    vlmLoading = false;
    
  } catch (error) {
    console.error('VLM loading error:', error);
    vlmLoading = false;
    vlmEnabled = false;
    
    updateAIStatus(`AI Error: ${error.message}`, 'error');
    document.getElementById('ai-description-toggle').checked = false;
  }
}

function captureVideoFrame(videoElement) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Use a reasonable size for the VLM
  const maxSize = 384;
  const scale = Math.min(maxSize / videoElement.videoWidth, maxSize / videoElement.videoHeight, 1);
  
  canvas.width = Math.round(videoElement.videoWidth * scale);
  canvas.height = Math.round(videoElement.videoHeight * scale);
  
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  
  // Return as base64 data URL
  return canvas.toDataURL('image/jpeg', 0.8);
}

async function generateDescription() {
  if (!vlmPipeline || !monitorVideo || !vlmEnabled) return;
  
  try {
    // Capture frame from video
    const imageData = captureVideoFrame(monitorVideo);
    
    // Generate caption
    const result = await vlmPipeline(imageData);
    
    // Extract the generated text
    let description = 'Unable to describe scene.';
    if (result && Array.isArray(result) && result[0]?.generated_text) {
      description = result[0].generated_text;
    } else if (result?.generated_text) {
      description = result.generated_text;
    }
    
    // Capitalize first letter and add period if missing
    description = description.charAt(0).toUpperCase() + description.slice(1);
    if (!description.endsWith('.') && !description.endsWith('!') && !description.endsWith('?')) {
      description += '.';
    }
    
    showAIDescription(description);
    
  } catch (error) {
    console.error('VLM inference error:', error);
    showAIDescription('<span class="ai-placeholder">Error processing frame. Retrying...</span>');
  }
}

function startVLMCapture() {
  if (vlmCaptureInterval) return;
  
  // Generate initial description after a short delay
  setTimeout(() => {
    if (vlmEnabled) generateDescription();
  }, 500);
  
  // Set up interval for periodic captures
  vlmCaptureInterval = setInterval(() => {
    if (vlmEnabled) generateDescription();
  }, VLM_CAPTURE_INTERVAL);
}

function stopVLMCapture() {
  if (vlmCaptureInterval) {
    clearInterval(vlmCaptureInterval);
    vlmCaptureInterval = null;
  }
}

// ============================================
// Liveness Detection Functions
// ============================================

// Calculate Eye Aspect Ratio (EAR)
function getEyeAspectRatio(landmarks, eyeIndices) {
  // eyeIndices should be [p1, p2, p3, p4, p5, p6]
  const p1 = landmarks[eyeIndices[0]];
  const p2 = landmarks[eyeIndices[1]];
  const p3 = landmarks[eyeIndices[2]];
  const p4 = landmarks[eyeIndices[3]];
  const p5 = landmarks[eyeIndices[4]];
  const p6 = landmarks[eyeIndices[5]];
  
  // Vertical distances
  const v1 = Math.sqrt(Math.pow(p2.x - p6.x, 2) + Math.pow(p2.y - p6.y, 2));
  const v2 = Math.sqrt(Math.pow(p3.x - p5.x, 2) + Math.pow(p3.y - p5.y, 2));
  
  // Horizontal distance
  const h = Math.sqrt(Math.pow(p1.x - p4.x, 2) + Math.pow(p1.y - p4.y, 2));
  
  return (v1 + v2) / (2 * h);
}

// Check if person is blinking
let currentEAR = 0; // For debug display

function detectBlink(landmarks) {
  // Left eye: landmarks 36-41
  const leftEye = [36, 37, 38, 39, 40, 41];
  // Right eye: landmarks 42-47
  const rightEye = [42, 43, 44, 45, 46, 47];
  
  const leftEAR = getEyeAspectRatio(landmarks, leftEye);
  const rightEAR = getEyeAspectRatio(landmarks, rightEye);
  const avgEAR = (leftEAR + rightEAR) / 2;
  
  currentEAR = avgEAR; // Store for display
  
  const BLINK_THRESHOLD = 0.28; // Increased for easier detection
  
  // Detect blink transition (open -> closed -> open)
  if (lastEAR > BLINK_THRESHOLD && avgEAR <= BLINK_THRESHOLD) {
    eyesWereClosed = true;
  } else if (eyesWereClosed && avgEAR > BLINK_THRESHOLD) {
    // Blink completed
    eyesWereClosed = false;
    lastEAR = avgEAR;
    return true;
  }
  
  lastEAR = avgEAR;
  return false;
}

// Detect head turn
function detectHeadTurn(landmarks, direction) {
  // landmarks[30] = nose tip
  // landmarks[2] = left face edge
  // landmarks[14] = right face edge
  const noseX = landmarks[30].x;
  const leftFace = landmarks[2].x;
  const rightFace = landmarks[14].x;
  const faceWidth = rightFace - leftFace;
  
  const relativeNoseX = (noseX - leftFace) / faceWidth;
  
  // relativeNoseX < 0.4 = turned right
  // relativeNoseX > 0.6 = turned left
  // 0.4-0.6 = facing forward
  
  if (direction === 'left' && relativeNoseX > 0.6) {
    return true;
  } else if (direction === 'right' && relativeNoseX < 0.4) {
    return true;
  }
  
  return false;
}

// Detect smile
let currentSmileScore = 0; // For debug display

function detectSmile(landmarks) {
  // Mouth landmarks for smile detection
  // Mouth corners: 48 (left), 54 (right)
  // Upper lip center: 51
  // Lower lip center: 57
  // Mouth center Y: average of 51 and 57
  
  const leftCorner = landmarks[48];
  const rightCorner = landmarks[54];
  const upperLipCenter = landmarks[51];
  const lowerLipCenter = landmarks[57];
  
  // Mouth center Y position
  const mouthCenterY = (upperLipCenter.y + lowerLipCenter.y) / 2;
  
  // When smiling, corners rise above mouth center (lower Y value)
  // Calculate how much corners are raised relative to mouth width
  const mouthWidth = Math.sqrt(
    Math.pow(leftCorner.x - rightCorner.x, 2) + Math.pow(leftCorner.y - rightCorner.y, 2)
  );
  
  // Positive value = corners are below center (neutral/frown)
  // Negative value = corners are above center (smile)
  const leftCornerRise = mouthCenterY - leftCorner.y;
  const rightCornerRise = mouthCenterY - rightCorner.y;
  const avgCornerRise = (leftCornerRise + rightCornerRise) / 2;
  
  // Normalize by mouth width
  const smileScore = avgCornerRise / mouthWidth;
  
  currentSmileScore = smileScore;
  
  // Smile score > 0.05 indicates corners are raised (smiling)
  return smileScore > 0.05;
}

// Shuffle array
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Start liveness check
async function startLivenessCheck() {
  const startBtn = document.getElementById('start-liveness-btn');
  const challengeEl = document.getElementById('liveness-challenge');
  const progressEl = document.getElementById('liveness-progress');
  const resultEl = document.getElementById('liveness-result');
  
  // Reset state
  currentChallenge = 0;
  blinkCount = 0;
  lastEAR = 0;
  eyesWereClosed = false;
  
  // Generate random challenges
  const turnDirection = Math.random() > 0.5 ? 'left' : 'right';
  challenges = shuffleArray([
    { type: 'blink', title: 'Blink Twice', description: 'Look at the camera and blink two times' },
    { type: 'turn', title: `Turn ${turnDirection === 'left' ? 'Left' : 'Right'}`, description: `Turn your head to the ${turnDirection}`, direction: turnDirection },
    { type: 'smile', title: 'Smile', description: 'Give us a natural smile' }
  ]);
  
  // Show UI
  startBtn.classList.add('hidden');
  challengeEl.classList.remove('hidden');
  progressEl.classList.remove('hidden');
  resultEl.classList.add('hidden');
  
  // Reset progress steps
  progressEl.querySelectorAll('.step').forEach(step => {
    step.className = 'step';
  });
  
  isLivenessCheck = true;
  
  // Start first challenge
  startNextChallenge();
}

// Start next challenge
function startNextChallenge() {
  if (currentChallenge >= challenges.length) {
    // All challenges completed
    livenessSuccess();
    return;
  }
  
  const challenge = challenges[currentChallenge];
  const titleEl = document.getElementById('challenge-title');
  const descEl = document.getElementById('challenge-description');
  const progressEl = document.getElementById('liveness-progress');
  
  // Update UI
  titleEl.textContent = challenge.title;
  descEl.textContent = challenge.description;
  
  // Update progress
  progressEl.querySelectorAll('.step').forEach((step, index) => {
    if (index < currentChallenge) {
      step.className = 'step completed';
    } else if (index === currentChallenge) {
      step.className = 'step active';
    } else {
      step.className = 'step';
    }
  });
  
  // Reset challenge-specific state
  if (challenge.type === 'blink') {
    blinkCount = 0;
  }
  
  // Set timeout for challenge (5 seconds)
  if (challengeTimeout) clearTimeout(challengeTimeout);
  challengeTimeout = setTimeout(() => {
    livenessFailed(`Challenge "${challenge.title}" timed out`);
  }, 5000);
}

// Check current challenge
function checkChallenge(landmarks) {
  if (!isLivenessCheck || currentChallenge >= challenges.length) return;
  
  const challenge = challenges[currentChallenge];
  let success = false;
  
  switch (challenge.type) {
    case 'blink':
      if (detectBlink(landmarks)) {
        blinkCount++;
        if (blinkCount >= 2) {
          success = true;
        }
      }
      // Show EAR value and blink count for debugging
      const descEl = document.getElementById('challenge-description');
      descEl.textContent = `${challenge.description} (Blinks: ${blinkCount}/2, EAR: ${currentEAR.toFixed(2)})`;
      break;
      
    case 'turn':
      if (detectHeadTurn(landmarks, challenge.direction)) {
        success = true;
      }
      break;
      
    case 'smile':
      if (detectSmile(landmarks)) {
        success = true;
      }
      // Show smile score for debugging
      const smileDescEl = document.getElementById('challenge-description');
      smileDescEl.textContent = `${challenge.description} (Score: ${currentSmileScore.toFixed(3)}${currentSmileScore > 0.05 ? ' ✓' : ''})`;
      break;
  }
  
  if (success) {
    clearTimeout(challengeTimeout);
    currentChallenge++;
    startNextChallenge();
  }
}

// Liveness success
function livenessSuccess() {
  isLivenessCheck = false;
  
  const challengeEl = document.getElementById('liveness-challenge');
  const resultEl = document.getElementById('liveness-result');
  const startBtn = document.getElementById('start-liveness-btn');
  const progressEl = document.getElementById('liveness-progress');
  
  challengeEl.classList.add('hidden');
  progressEl.classList.add('hidden');
  resultEl.classList.remove('hidden');
  resultEl.className = 'liveness-result success';
  resultEl.innerHTML = `
    <div class="result-icon">✓</div>
    <div class="result-title">Liveness Verified</div>
    <div class="result-message">You are a real person!</div>
  `;
  
  startBtn.textContent = 'Try Again';
  startBtn.classList.remove('hidden');
}

// Liveness failed
function livenessFailed(reason) {
  isLivenessCheck = false;
  
  const challengeEl = document.getElementById('liveness-challenge');
  const resultEl = document.getElementById('liveness-result');
  const startBtn = document.getElementById('start-liveness-btn');
  const progressEl = document.getElementById('liveness-progress');
  
  challengeEl.classList.add('hidden');
  progressEl.classList.add('hidden');
  resultEl.classList.remove('hidden');
  resultEl.className = 'liveness-result failed';
  resultEl.innerHTML = `
    <div class="result-icon">✗</div>
    <div class="result-title">Verification Failed</div>
    <div class="result-message">${reason || 'Please try again'}</div>
  `;
  
  startBtn.textContent = 'Try Again';
  startBtn.classList.remove('hidden');
}

// ============================================
// Camera Functions
// ============================================

async function startCamera(videoElement) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    });
    videoElement.srcObject = stream;
    return true;
  } catch (error) {
    console.error('Camera error:', error);
    throw new Error('Camera access denied. Please allow camera permissions.');
  }
}

function stopCamera(videoElement) {
  if (videoElement && videoElement.srcObject) {
    videoElement.srcObject.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
  }
}

// ============================================
// Face Detection & Recognition
// ============================================

async function detectFace(videoElement, canvasElement) {
  if (!modelsLoaded) return null;
  
  const detection = await faceapi
    .detectSingleFace(videoElement, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();
  
  if (canvasElement && detection) {
    // Draw detection on canvas
    const displaySize = { width: videoElement.videoWidth, height: videoElement.videoHeight };
    faceapi.matchDimensions(canvasElement, displaySize);
    const resizedDetection = faceapi.resizeResults(detection, displaySize);
    
    const ctx = canvasElement.getContext('2d');
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw box
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    const box = resizedDetection.detection.box;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    
    // Draw landmarks
    ctx.fillStyle = '#10b981';
    resizedDetection.landmarks.positions.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  
  return detection;
}

function compareFaces(descriptor1, descriptor2) {
  // Euclidean distance
  const distance = faceapi.euclideanDistance(descriptor1, descriptor2);
  return {
    distance,
    match: distance < MATCH_THRESHOLD
  };
}

// ============================================
// Storage Functions
// ============================================

function saveFaceDescriptor(descriptor) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(descriptor)));
  return true;
}

function getFaceDescriptor() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return null;
  return new Float32Array(JSON.parse(stored));
}

function hasRegisteredFace() {
  return localStorage.getItem(STORAGE_KEY) !== null;
}

// ============================================
// Register Page Logic
// ============================================

async function initRegisterPage() {
  registerVideo = document.getElementById('register-video');
  const registerCanvas = document.getElementById('register-canvas');
  const registerBtn = document.getElementById('register-btn');
  const registerStatus = document.getElementById('register-status');
  
  try {
    registerStatus.className = 'status-message info';
    registerStatus.textContent = 'Starting camera...';
    
    await startCamera(registerVideo);
    
    registerStatus.textContent = 'Camera ready. Position your face and click Register.';
    registerBtn.disabled = false;
    
    // Live detection preview
    setInterval(async () => {
      const detection = await detectFace(registerVideo, registerCanvas);
      if (detection) {
        registerBtn.disabled = false;
      }
    }, 200);
    
  } catch (error) {
    registerStatus.className = 'status-message error';
    registerStatus.textContent = error.message;
  }
  
  // Register button handler
  registerBtn.addEventListener('click', async () => {
    registerBtn.disabled = true;
    registerStatus.className = 'status-message info';
    registerStatus.textContent = 'Detecting face...';
    
    try {
      const detection = await detectFace(registerVideo, registerCanvas);
      
      if (!detection) {
        throw new Error('No face detected. Please position your face in the frame.');
      }
      
      // Save descriptor
      saveFaceDescriptor(detection.descriptor);
      
      registerStatus.className = 'status-message success';
      registerStatus.textContent = '✓ Face registered successfully! You can now use the Monitor page.';
      
      setTimeout(() => {
        registerBtn.disabled = false;
      }, 2000);
      
    } catch (error) {
      registerStatus.className = 'status-message error';
      registerStatus.textContent = error.message;
      registerBtn.disabled = false;
    }
  });
}

// ============================================
// Monitor Page Logic
// ============================================

async function initMonitorPage() {
  monitorVideo = document.getElementById('monitor-video');
  const monitorCanvas = document.getElementById('monitor-canvas');
  const matchResult = document.getElementById('match-result');
  const aiToggle = document.getElementById('ai-description-toggle');
  
  // Check if face is registered
  if (!hasRegisteredFace()) {
    matchResult.innerHTML = `
      <div class="match-status" style="color: var(--accent-warning);">No Face Registered</div>
      <div class="match-confidence">Please register your face first on the Register page.</div>
    `;
    return;
  }
  
  try {
    await startCamera(monitorVideo);
    isMonitoring = true;
    
    const savedDescriptor = getFaceDescriptor();
    
    // Start monitoring loop
    monitorInterval = setInterval(async () => {
      if (!isMonitoring) return;
      
      const detection = await detectFace(monitorVideo, monitorCanvas);
      
      if (!detection) {
        matchResult.className = 'match-result';
        matchResult.innerHTML = `
          <div class="match-status">No Face Detected</div>
          <div class="match-confidence">Position your face in the camera</div>
        `;
        return;
      }
      
      // Compare with saved face
      const result = compareFaces(savedDescriptor, detection.descriptor);
      
      if (result.match) {
        matchResult.className = 'match-result match';
        matchResult.innerHTML = `
          <div class="match-status">✓ MATCH</div>
          <div class="match-confidence">Distance: ${result.distance.toFixed(3)} (threshold: ${MATCH_THRESHOLD})</div>
        `;
      } else {
        matchResult.className = 'match-result alert';
        matchResult.innerHTML = `
          <div class="match-status">⚠ ALERT</div>
          <div class="match-confidence">Face does not match (distance: ${result.distance.toFixed(3)})</div>
        `;
      }
    }, 300);
    
    // Setup AI toggle handler
    aiToggle.addEventListener('change', async (e) => {
      vlmEnabled = e.target.checked;
      
      if (vlmEnabled) {
        // Load VLM if not loaded
        if (!vlmPipeline) {
          await loadVLM();
        }
        
        if (vlmPipeline) {
          startVLMCapture();
        }
      } else {
        stopVLMCapture();
        document.getElementById('ai-description-container').classList.add('hidden');
        hideAIStatus();
      }
    });
    
  } catch (error) {
    matchResult.innerHTML = `
      <div class="match-status" style="color: var(--accent-danger);">Camera Error</div>
      <div class="match-confidence">${error.message}</div>
    `;
  }
}

function stopMonitorPage() {
  isMonitoring = false;
  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
  }
  stopVLMCapture();
  stopCamera(monitorVideo);
}

// ============================================
// Liveness Page Logic
// ============================================

async function initLivenessPage() {
  livenessVideo = document.getElementById('liveness-video');
  livenessCanvas = document.getElementById('liveness-canvas');
  const startBtn = document.getElementById('start-liveness-btn');
  
  try {
    await startCamera(livenessVideo);
    
    // Start live detection preview
    livenessInterval = setInterval(async () => {
      const detection = await detectFace(livenessVideo, livenessCanvas);
      
      // Check challenge if active
      if (detection && isLivenessCheck) {
        const landmarks = detection.landmarks.positions;
        checkChallenge(landmarks);
      }
    }, 100); // Check every 100ms for smoother detection
    
    // Setup start button
    startBtn.addEventListener('click', () => {
      if (!isLivenessCheck) {
        startLivenessCheck();
      }
    });
    
  } catch (error) {
    const resultEl = document.getElementById('liveness-result');
    resultEl.classList.remove('hidden');
    resultEl.className = 'liveness-result failed';
    resultEl.innerHTML = `
      <div class="result-icon">✗</div>
      <div class="result-title">Camera Error</div>
      <div class="result-message">${error.message}</div>
    `;
    startBtn.disabled = true;
  }
}

function stopLivenessPage() {
  isLivenessCheck = false;
  if (livenessInterval) {
    clearInterval(livenessInterval);
    livenessInterval = null;
  }
  if (challengeTimeout) {
    clearTimeout(challengeTimeout);
    challengeTimeout = null;
  }
  stopCamera(livenessVideo);
}

// ============================================
// Router
// ============================================

function getRoute() {
  return window.location.hash.slice(1) || '/';
}

function navigateTo(route) {
  window.location.hash = route;
}

async function handleRoute() {
  const route = getRoute();
  
  // Update nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    const linkRoute = link.getAttribute('data-route');
    if (linkRoute === route) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
  
  // Hide all pages
  document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
  
  // Stop any running monitors
  stopMonitorPage();
  stopLivenessPage();
  
  // Show appropriate page
  if (route === '/monitor') {
    document.getElementById('monitor-page').classList.add('active');
    await initMonitorPage();
  } else if (route === '/liveness') {
    document.getElementById('liveness-page').classList.add('active');
    await initLivenessPage();
  } else {
    document.getElementById('register-page').classList.add('active');
    await initRegisterPage();
  }
}

// ============================================
// App Initialization
// ============================================

async function initApp() {
  // Load models with progress
  try {
    await loadModelsWithProgress();
  } catch (error) {
    console.error('Failed to load models:', error);
    document.getElementById('progress-detail').textContent = 
      `Failed to load models: ${error.message}. Please check your internet connection and refresh.`;
    return;
  }
  
  // Hide loading screen, show main app
  document.getElementById('loading-screen').classList.remove('active');
  document.getElementById('main-app').classList.remove('hidden');
  
  // Setup router
  window.addEventListener('hashchange', handleRoute);
  
  // Handle nav links
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const route = link.getAttribute('data-route');
      navigateTo(route);
    });
  });
  
  // Initial route
  if (!window.location.hash) {
    navigateTo('/');
  } else {
    await handleRoute();
  }
}

// Start app
initApp().catch(console.error);
