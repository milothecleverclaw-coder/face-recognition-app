import * as faceapi from 'face-api.js';

// Configuration
const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
const STORAGE_KEY = 'registered_face_descriptor';
const MATCH_THRESHOLD = 0.6; // Lower = stricter matching

// State
let modelsLoaded = false;
let registerVideo = null;
let monitorVideo = null;
let isMonitoring = false;
let monitorInterval = null;

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
  stopCamera(monitorVideo);
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
  
  // Show appropriate page
  if (route === '/monitor') {
    document.getElementById('monitor-page').classList.add('active');
    await initMonitorPage();
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
