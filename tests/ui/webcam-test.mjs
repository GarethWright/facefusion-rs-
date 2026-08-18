// See README.md. Run against a FaceFusion.Ui already listening on 127.0.0.1:7860.
import { chromium } from 'playwright';

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
const page = await browser.newPage();
const fails = [];
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  — ' + extra : ''}`); if (!ok) fails.push(name); };
page.on('console', m => { if (m.type() === 'error') console.log('BROWSER ERROR:', m.text()); });

await page.goto('http://127.0.0.1:7860/webcam', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

check('webcam layout renders', await page.locator('h2:text-is("WEBCAM")').count() === 1);
check('stream panel renders', await page.locator('h2:text-is("STREAM")').count() === 1);
check('no camera detected in this container', (await page.locator('#ff-webcam-device').textContent()).includes('none detected'));

// Use a video file as the camera source — Python's own get_remote_camera_capture accepts one.
await page.locator('#ff-webcam-remote').fill(process.env.WEBCAM_TEST_SOURCE ?? '/tmp/facefusion-test-examples/target-240p.mp4');
await page.locator('#ff-webcam-remote').blur();
await page.locator('#ff-webcam-resolution').selectOption('320x240');

// frame_colorizer streams fast enough on CPU to see frames within the timeout.
const processors = page.locator('section.ff-panel', { has: page.locator('h2:text-is("PROCESSORS")') });
await processors.locator('label.ff-checkbox', { hasText: 'face_swapper' }).locator('input').uncheck();
await processors.locator('label.ff-checkbox', { hasText: 'frame_colorizer' }).locator('input').check();
await page.waitForTimeout(700);

await page.locator('button:text("START")').click();
await page.waitForSelector('#ff-webcam-image', { timeout: 180000 });
check('a streamed frame reached the browser', true);

const first = await page.locator('#ff-webcam-image').getAttribute('src');
check('the frame is a jpeg data uri', first.startsWith('data:image/jpeg;base64,'), `${first.length} chars`);

// A second, different frame proves it is a stream rather than one still.
await page.waitForFunction(
    prev => document.querySelector('#ff-webcam-image')?.getAttribute('src') !== prev,
    first, { timeout: 180000 });
check('the stream advances to a new frame', true);

check('the stop button is shown while streaming', await page.locator('button:text("STOP")').count() === 1);
await page.locator('button:text("STOP")').click();
await page.waitForFunction(() => document.querySelector('button')?.textContent?.includes('START'), { timeout: 60000 });
check('stopping restores the start button', true);

await page.screenshot({ path: '/tmp/pv/webcam.png', fullPage: true });
await browser.close();
console.log(fails.length === 0 ? '\nALL CHECKS PASSED' : `\n${fails.length} CHECK(S) FAILED: ${fails.join(', ')}`);
process.exit(fails.length === 0 ? 0 : 1);
