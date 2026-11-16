import init, { start_doom, toggle_msaa, set_msaa } from './pkg/room.js';

window.toggle_msaa = toggle_msaa;
window.set_msaa = set_msaa;

const canvas = document.getElementById('doom-canvas');
const status = document.getElementById('status');

canvas.addEventListener('click', () => canvas.focus());

try {
  await init();
  start_doom('doom-canvas');
  canvas.focus();
} catch (err) {
  status.textContent = err instanceof Error ? err.message : String(err);
  console.error(err);
}
