import init, { start_doom, toggle_msaa, set_msaa, is_msaa_enabled, toggle_smaa, set_smaa, is_smaa_enabled, set_ssrc, is_ssrc_enabled, toggle_ssrc, set_ssrc_strength, get_ssrc_strength, clear_visible, set_normal_strength, get_normal_strength, set_roughness_scale, get_roughness_scale, set_specular_sheen, get_specular_sheen, set_bounce_boost, get_bounce_boost, set_baked_light, is_baked_light_enabled, toggle_baked_light, set_diminish, get_diminish, set_light_blend, get_light_blend, set_voxel_blend, get_voxel_blend, toggle_pause, is_paused } from './pkg/room.js';

window.toggle_pause = toggle_pause;
window.is_paused = is_paused;
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

const canvas = document.getElementById('doom-canvas');
const status = document.getElementById('status');

window.addEventListener('keydown', (e) => {
  if (e.key === 'F5' || e.key === ']') {
    e.preventDefault();
  } else if (e.key === 'F6' || e.key === '[') {
    e.preventDefault();
  } else if (e.key === 'F8') {
    clear_visible();
  }
});

canvas.addEventListener('click', () => canvas.focus());

try {
  await init();
  start_doom('doom-canvas');
  canvas.focus();
} catch (err) {
  status.textContent = err instanceof Error ? err.message : String(err);
  console.error(err);
}
