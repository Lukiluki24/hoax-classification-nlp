document.addEventListener('DOMContentLoaded', async () => {
  const statusDiv = document.getElementById('server-status');
  const toggleBtn = document.getElementById('toggle-btn');

  // Cek apakah server FastAPI uvicorn di port 8000 sedang aktif
  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "tes koneksi" })
    });

    if (response.ok || response.status === 200 || response.status === 400) {
      statusDiv.textContent = "API BACKEND CONNECTED";
      statusDiv.style.color = "#28a745";
    }
  } catch (error) {
    statusDiv.textContent = "API BACKEND DISCONNECTED";
    statusDiv.style.color = "#e50914";
  }

  // Logika Tombol Nyala/Mati
  chrome.storage.local.get(['isShieldActive'], function(result) {
    let isActive = result.isShieldActive !== false; 
    updateButtonUI(isActive);
  });

  toggleBtn.addEventListener('click', () => {
    chrome.storage.local.get(['isShieldActive'], function(result) {
      let isActive = result.isShieldActive !== false;
      let newState = !isActive;
      
      chrome.storage.local.set({ isShieldActive: newState }, function() {
        updateButtonUI(newState);
        
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
          if(tabs[0]) {
            chrome.tabs.reload(tabs[0].id);
          }
        });
      });
    });
  });

  function updateButtonUI(isActive) {
    if (isActive) {
      toggleBtn.textContent = "TURN OFF";
      toggleBtn.className = "btn-disable";
    } else {
      toggleBtn.textContent = "TURN ON";
      toggleBtn.className = "btn-enable";
    }
  }
});