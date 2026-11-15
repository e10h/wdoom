import init, { start_doom } from './pkg/room.js';

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
