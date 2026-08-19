// See README.md. Run against a FaceFusion.Ui already listening on 127.0.0.1:7860.
import { chromium } from 'playwright';
import fs from 'fs';

const OUT = process.env.UI_TEST_OUTPUT ?? '/tmp/facefusion-ui-test-run.mp4';
if (fs.existsSync(OUT)) fs.unlinkSync(OUT);

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox'] });
const page = await browser.newPage();
const fails = [];
const check = (name, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  — ' + extra : ''}`); if (!ok) fails.push(name); };

page.on('console', m => { if (m.type() === 'error') console.log('BROWSER ERROR:', m.text()); });

await page.goto('http://127.0.0.1:7860/', { waitUntil: 'networkidle' });
// Blazor Server needs the circuit up before any handler runs.
await page.waitForFunction(() => !document.querySelector('#components-reconnect-modal'));
await page.waitForTimeout(1500);

const panel = name => page.locator('section.ff-panel', { has: page.locator(`h2:text-is("${name}")`) });

check('face swapper panel shown by default', await panel('FACE SWAPPER').count() === 1);
check('frame colorizer panel hidden by default', await panel('FRAME COLORIZER').count() === 0);

// Cross-component wiring: ticking a processor must reveal its options block.
await page.locator('section.ff-panel', { has: page.locator('h2:text-is("PROCESSORS")') })
    .locator('label.ff-checkbox', { hasText: 'frame_colorizer' }).locator('input').check();
await page.waitForTimeout(700);
check('ticking frame_colorizer reveals its options', await panel('FRAME COLORIZER').count() === 1);

// ...and unticking hides it again.
await page.locator('section.ff-panel', { has: page.locator('h2:text-is("PROCESSORS")') })
    .locator('label.ff-checkbox', { hasText: 'frame_colorizer' }).locator('input').uncheck();
await page.waitForTimeout(700);
check('unticking frame_colorizer hides it again', await panel('FRAME COLORIZER').count() === 0);

// The face-swapper model change must reset pixel boost to the new model's first choice.
const pixelBoostBefore = await page.locator('#ff-face-swapper-pixel-boost').inputValue();
await page.locator('#ff-face-swapper-model').selectOption('inswapper_128');
await page.waitForTimeout(700);
const pixelBoostAfter = await page.locator('#ff-face-swapper-pixel-boost').inputValue();
check('changing the model resets pixel boost', pixelBoostAfter === '128x128',
    `${pixelBoostBefore} -> ${pixelBoostAfter}`);
await page.locator('#ff-face-swapper-model').selectOption('hyperswap_1a_256');
await page.waitForTimeout(500);

// Reference-face controls appear only in reference mode.
const selector = panel('FACE SELECTOR');
check('reference controls shown in reference mode',
    await selector.locator('label:text-is("REFERENCE FACE POSITION")').count() === 1);
await selector.locator('select').first().selectOption('one');
await page.waitForTimeout(700);
check('reference controls hidden in one mode',
    await selector.locator('label:text-is("REFERENCE FACE POSITION")').count() === 0);
await selector.locator('select').first().selectOption('reference');
await page.waitForTimeout(500);

// Now a real run.
const setText = async (panelName, labelText, value) => {
    const field = panel(panelName).locator('.ff-field', { has: page.locator(`label:text-is("${labelText}")`) });
    await field.locator('input').fill(value);
    await field.locator('input').blur();
    await page.waitForTimeout(400);
};

await setText('TARGET', 'TARGET PATH', '/tmp/facefusion-test-examples/target-240p.mp4');
await setText('OUTPUT', 'OUTPUT PATH', OUT);
await setText('TRIM FRAME', 'TRIM FRAME START', '0');
await setText('TRIM FRAME', 'TRIM FRAME END', '4');
await panel('PROCESSORS').locator('label.ff-checkbox', { hasText: 'face_swapper' }).locator('input').uncheck();
await panel('PROCESSORS').locator('label.ff-checkbox', { hasText: 'frame_colorizer' }).locator('input').check();
await page.waitForTimeout(700);

check('target path reported as found', (await panel('TARGET').locator('.ff-ok').count()) === 1);

await page.locator('button:text("START")').click();
await page.waitForFunction(
    () => document.querySelector('.ff-terminal')?.textContent?.includes('processing to video succeeded'),
    { timeout: 300000 });

check('run wrote the output file', fs.existsSync(OUT), OUT);
const terminal = await page.locator('.ff-terminal').textContent();
console.log('--- terminal ---\n' + terminal.trim() + '\n----------------');

// Preview.
await page.locator('button:text("RENDER PREVIEW")').click();
await page.waitForFunction(
    () => document.querySelector('img.ff-preview') !== null || document.querySelector('.ff-error') !== null,
    { timeout: 300000 });
check('preview rendered an image', await page.locator('img.ff-preview').count() === 1,
    await page.locator('img.ff-preview').count() === 1 ? '' : await page.locator('.ff-error').first().textContent());

await page.screenshot({ path: '/tmp/pv/ui.png', fullPage: true });
await browser.close();

console.log(fails.length === 0 ? '\nALL CHECKS PASSED' : `\n${fails.length} CHECK(S) FAILED: ${fails.join(', ')}`);
process.exit(fails.length === 0 ? 0 : 1);
