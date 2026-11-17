import init, { start_doom, toggle_msaa, set_msaa, is_msaa_enabled, toggle_smaa, set_smaa, is_smaa_enabled, clear_visible } from './pkg/room.js';

window.toggle_msaa = toggle_msaa;
window.set_msaa = set_msaa;
window.is_msaa_enabled = is_msaa_enabled;
window.toggle_smaa = toggle_smaa;
window.set_smaa = set_smaa;
window.is_smaa_enabled = is_smaa_enabled;
window.clear_visible = clear_visible;

const canvas = document.getElementById('doom-canvas');
const status = document.getElementById('status');
const btnMsaa = document.getElementById('btn-msaa');
const btnSmaa = document.getElementById('btn-smaa');
const btnClearVisible = document.getElementById('btn-clear-visible');

function updateButtonStates() {
  try {
    const msaaOn = is_msaa_enabled();
    btnMsaa.textContent = `MSAA (4×): ${msaaOn ? 'ON' : 'OFF'}`;
    btnMsaa.classList.toggle('active', msaaOn);

    const smaaOn = is_smaa_enabled();
    btnSmaa.textContent = `SMAA: ${smaaOn ? 'ON' : 'OFF'}`;
    btnSmaa.classList.toggle('active', smaaOn);
  } catch (_) {
    // Before doom/GPU is initialized
  }
}

if (btnMsaa) {
  btnMsaa.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle_msaa();
    updateButtonStates();
    canvas.focus();
  });
}

if (btnSmaa) {
  btnSmaa.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle_smaa();
    updateButtonStates();
    canvas.focus();
  });
}

if (btnClearVisible) {
  btnClearVisible.addEventListener('click', (e) => {
    e.stopPropagation();
    clear_visible();
    canvas.focus();
  });
}

window.addEventListener('keydown', (e) => {
  if (e.key === '0') {
    setTimeout(updateButtonStates, 10);
  } else if (e.key === '\\' || e.key === 'F7') {
    setTimeout(updateButtonStates, 10);
  } else if (e.key === 'F8') {
    clear_visible();
  }
});

canvas.addEventListener('click', () => canvas.focus());

try {
  await init();
  start_doom('doom-canvas');
  canvas.focus();
  setTimeout(updateButtonStates, 200);
} catch (err) {
  status.textContent = err instanceof Error ? err.message : String(err);
  console.error(err);
}
