import init, { start_doom, toggle_msaa, set_msaa, is_msaa_enabled, toggle_smaa, set_smaa, is_smaa_enabled, set_ssrc, is_ssrc_enabled, toggle_ssrc, set_ssrc_strength, get_ssrc_strength, clear_visible, set_normal_strength, get_normal_strength, set_roughness_scale, get_roughness_scale, set_specular_sheen, get_specular_sheen, set_bounce_boost, get_bounce_boost, set_baked_light, is_baked_light_enabled, toggle_baked_light, set_diminish, get_diminish, set_light_blend, get_light_blend, set_voxel_blend, get_voxel_blend, toggle_pause, is_paused, toggle_automap, is_automap_active, toggle_tex_scale, get_tex_scale, set_tex_scale } from './pkg/room.js';

window.toggle_pause = toggle_pause;
window.is_paused = is_paused;
window.toggle_automap = toggle_automap;
window.is_automap_active = is_automap_active;
window.toggle_msaa = toggle_msaa;
window.set_msaa = set_msaa;
window.is_msaa_enabled = is_msaa_enabled;
window.toggle_smaa = toggle_smaa;
window.set_smaa = set_smaa;
window.is_smaa_enabled = is_smaa_enabled;
window.set_ssrc = set_ssrc;
window.is_ssrc_enabled = is_ssrc_enabled;
window.toggle_ssrc = toggle_ssrc;
window.set_ssrc_strength = set_ssrc_strength;
window.get_ssrc_strength = get_ssrc_strength;
window.clear_visible = clear_visible;
window.set_normal_strength = set_normal_strength;
window.get_normal_strength = get_normal_strength;
window.set_roughness_scale = set_roughness_scale;
window.get_roughness_scale = get_roughness_scale;
window.set_specular_sheen = set_specular_sheen;
window.get_specular_sheen = get_specular_sheen;
window.set_bounce_boost = set_bounce_boost;
window.get_bounce_boost = get_bounce_boost;
window.set_baked_light = set_baked_light;
window.is_baked_light_enabled = is_baked_light_enabled;
window.toggle_baked_light = toggle_baked_light;
window.set_diminish = set_diminish;
window.get_diminish = get_diminish;
window.set_light_blend = set_light_blend;
window.get_light_blend = get_light_blend;
window.set_voxel_blend = set_voxel_blend;
window.get_voxel_blend = get_voxel_blend;
window.toggle_tex_scale = toggle_tex_scale;
window.get_tex_scale = get_tex_scale;
window.set_tex_scale = set_tex_scale;

const canvas = document.getElementById('doom-canvas');
const status = document.getElementById('status');

const btnBake = document.getElementById('btn-bake');
const btnMap = document.getElementById('btn-map');
const btnPause = document.getElementById('btn-pause');
const btnSsrc = document.getElementById('btn-ssrc');
const btnClearVisible = document.getElementById('btn-clear-visible');
const btnMsaa = document.getElementById('btn-msaa');
const btnSmaa = document.getElementById('btn-smaa');
const btnTexScale = document.getElementById('btn-tex-scale');
const btnTogglePanel = document.getElementById('btn-toggle-panel');
const controlsBody = document.getElementById('controls-body');

if (btnTogglePanel && controlsBody) {
  btnTogglePanel.addEventListener('click', (e) => {
    e.stopPropagation();
    const hidden = controlsBody.style.display === 'none';
    controlsBody.style.display = hidden ? '' : 'none';
    btnTogglePanel.textContent = hidden ? '−' : '+';
    btnTogglePanel.title = hidden ? 'Minimize controls' : 'Expand controls';
    canvas.focus();
  });
}

function updateButtonStates() {
  try {
    if (btnBake) {
      const on = is_baked_light_enabled();
      btnBake.textContent = `Bake: ${on ? 'ON' : 'OFF'}`;
      btnBake.classList.toggle('active', on);
    }
    if (btnMap) {
      const on = is_automap_active();
      btnMap.textContent = `Map: ${on ? 'ON' : 'OFF'}`;
      btnMap.classList.toggle('active', on);
    }
    if (btnPause) {
      const on = is_paused();
      btnPause.textContent = `Pause: ${on ? 'ON' : 'OFF'}`;
      btnPause.classList.toggle('active', on);
    }
    if (btnMsaa) {
      const on = is_msaa_enabled();
      btnMsaa.textContent = `MSAA: ${on ? 'ON' : 'OFF'}`;
      btnMsaa.classList.toggle('active', on);
    }
    if (btnSmaa) {
      const on = is_smaa_enabled();
      btnSmaa.textContent = `SMAA: ${on ? 'ON' : 'OFF'}`;
      btnSmaa.classList.toggle('active', on);
    }
    if (btnSsrc) {
      const on = is_ssrc_enabled();
      btnSsrc.textContent = `SSRC: ${on ? 'ON' : 'OFF'}`;
      btnSsrc.classList.toggle('active', on);
    }
    if (btnTexScale) {
      const scale = get_tex_scale();
      const isHalf = Math.abs(scale - 0.5) < 0.05;
      btnTexScale.textContent = `Tex: ${isHalf ? '50%' : '100%'}`;
      btnTexScale.classList.toggle('active', isHalf);
    }
  } catch (_) {}
}

if (btnBake) {
  btnBake.addEventListener('click', (e) => {
    e.stopPropagation();
    set_baked_light(!is_baked_light_enabled());
    updateButtonStates();
    canvas.focus();
  });
}

if (btnMap) {
  btnMap.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle_automap();
    updateButtonStates();
    canvas.focus();
  });
}

if (btnPause) {
  btnPause.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle_pause();
    updateButtonStates();
    canvas.focus();
  });
}

if (btnSsrc) {
  btnSsrc.addEventListener('click', (e) => {
    e.stopPropagation();
    set_ssrc(!is_ssrc_enabled());
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

if (btnTexScale) {
  btnTexScale.addEventListener('click', (e) => {
    e.stopPropagation();
    toggle_tex_scale();
    updateButtonStates();
    canvas.focus();
  });
}

function bindSlider(sliderId, valId, setter) {
  const slider = document.getElementById(sliderId);
  const valSpan = document.getElementById(valId);
  if (!slider) return;
  const update = () => {
    const v = parseFloat(slider.value);
    setter(v);
    if (valSpan) valSpan.textContent = v.toFixed(2);
  };
  slider.addEventListener('input', (e) => {
    e.stopPropagation();
    update();
  });
}

bindSlider('slider-blend', 'val-blend', set_light_blend);
bindSlider('slider-voxel-blend', 'val-voxel-blend', set_voxel_blend);
bindSlider('slider-diminish', 'val-diminish', set_diminish);
bindSlider('slider-bounce', 'val-bounce', set_bounce_boost);
bindSlider('slider-normal', 'val-normal', set_normal_strength);
bindSlider('slider-roughness', 'val-roughness', set_roughness_scale);
bindSlider('slider-ssrc', 'val-ssrc', set_ssrc_strength);

window.addEventListener('keydown', (e) => {
  if (e.key === 'Tab') {
    e.preventDefault();
    setTimeout(updateButtonStates, 50);
  } else if (e.key === 'F5' || e.key === ']') {
    e.preventDefault();
    setTimeout(updateButtonStates, 10);
  } else if (e.key === 'F6' || e.key === '[') {
    e.preventDefault();
    setTimeout(updateButtonStates, 10);
  } else if (e.key === '0') {
    setTimeout(updateButtonStates, 10);
  } else if (e.key === '\\' || e.key === 'F7') {
    setTimeout(updateButtonStates, 10);
  } else if (e.key === 'F8') {
    clear_visible();
  } else if (e.key === 'p' || e.key === 'P' || e.key === 'm' || e.key === 'M') {
    setTimeout(updateButtonStates, 10);
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
