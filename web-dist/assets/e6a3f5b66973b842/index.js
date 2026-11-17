import init, { start_doom, toggle_msaa, set_msaa, is_msaa_enabled, toggle_smaa, set_smaa, is_smaa_enabled, clear_visible, set_normal_strength, get_normal_strength, set_roughness_scale, get_roughness_scale, set_specular_sheen, get_specular_sheen, set_ssrc, is_ssrc_enabled, set_ssrc_strength, get_ssrc_strength } from './pkg/room.js';

window.toggle_msaa = toggle_msaa;
window.set_msaa = set_msaa;
window.is_msaa_enabled = is_msaa_enabled;
window.toggle_smaa = toggle_smaa;
window.set_smaa = set_smaa;
window.is_smaa_enabled = is_smaa_enabled;
window.clear_visible = clear_visible;
window.set_normal_strength = set_normal_strength;
window.get_normal_strength = get_normal_strength;
window.set_roughness_scale = set_roughness_scale;
window.get_roughness_scale = get_roughness_scale;
window.set_specular_sheen = set_specular_sheen;
window.get_specular_sheen = get_specular_sheen;
window.set_ssrc = set_ssrc;
window.is_ssrc_enabled = is_ssrc_enabled;
window.set_ssrc_strength = set_ssrc_strength;
window.get_ssrc_strength = get_ssrc_strength;

const canvas = document.getElementById('doom-canvas');
const status = document.getElementById('status');
const btnBake = document.getElementById('btn-bake');
const btnMsaa = document.getElementById('btn-msaa');
const btnSmaa = document.getElementById('btn-smaa');
const btnClearVisible = document.getElementById('btn-clear-visible');
const sliderBounce = document.getElementById('slider-bounce');
const sliderNormal = document.getElementById('slider-normal');
const sliderRoughness = document.getElementById('slider-roughness');

function updateButtonStates() {
  try {
    if (btnBake) {
      const bakeOn = is_baked_light_enabled();
      btnBake.textContent = `Bake: ${bakeOn ? 'ON' : 'OFF'}`;
      btnBake.classList.toggle('active', bakeOn);
    }

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

if (btnBake) {
  btnBake.addEventListener('click', (e) => {
    e.stopPropagation();
    const now = !is_baked_light_enabled();
    set_baked_light(now);
    updateButtonStates();
    canvas.focus();
  });
}

if (sliderBounce) {
  sliderBounce.addEventListener('input', (e) => {
    e.stopPropagation();
    set_bounce_boost(parseFloat(e.target.value));
  });
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

if (sliderNormal) {
  sliderNormal.addEventListener('input', (e) => {
    e.stopPropagation();
    set_normal_strength(parseFloat(e.target.value));
  });
}

if (sliderRoughness) {
  sliderRoughness.addEventListener('input', (e) => {
    e.stopPropagation();
    set_roughness_scale(parseFloat(e.target.value));
  });
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'F6' || e.key === '[') {
    e.preventDefault();
    setTimeout(updateButtonStates, 10);
  } else if (e.key === '0') {
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
