import json
import time
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright
from agents import TestGenerationAgent, VisualVerificationAgent, ReportingAgent
from measurement_tools import draw_length, viewport_box
from test_data import load_test_cases

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
OUTPUT = ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT.mkdir(parents=True, exist_ok=True)

def log(message):
    print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)

def dismiss_tutorial(page):
    try:
        locator = page.get_by_text('Skip all', exact=True)
        if locator.count() and locator.first.is_visible():
            locator.first.click(timeout=3000)
    except Exception:
        pass
    try:
        locator = page.get_by_text('Confirm and hide', exact=True)
        if locator.count() and locator.first.is_visible():
            locator.first.click(timeout=3000)
    except Exception:
        pass

def discover(page):
    return page.evaluate('''() => {
      const canvas = [...document.querySelectorAll('canvas')].map((e,i) => {
        const r=e.getBoundingClientRect();return {index:i,x:Math.round(r.x),y:Math.round(r.y),width:Math.round(r.width),height:Math.round(r.height),visible:r.width>100&&r.height>100};
      });
      const controls=[...document.querySelectorAll('button,[role="button"],[title],[aria-label]')].slice(0,120).map((e)=>{
        const r=e.getBoundingClientRect();return {tag:e.tagName,text:(e.innerText||'').trim().slice(0,60),title:e.getAttribute('title'),label:e.getAttribute('aria-label'),x:Math.round(r.x),y:Math.round(r.y)};
      });
      return {url:location.href,resolution:{width:innerWidth,height:innerHeight,devicePixelRatio:devicePixelRatio},canvas,controls};
    }''')

def reload_clean_viewer(page, viewer_url):
    log('Restoring DICOM viewer')
    page.goto(viewer_url, wait_until='domcontentloaded', timeout=60000)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        dismiss_tutorial(page)
        evidence = discover(page)
        if any(c['visible'] and c['width'] >= 250 and c['height'] >= 250
               for c in evidence['canvas']):
            log('DICOM viewport detected')
            page.wait_for_timeout(4000)
            return
        page.wait_for_timeout(1500)
    page.screenshot(path=str(OUTPUT / 'viewer_restore_error.png'))
    raise RuntimeError('DICOM viewport not detected within 90 seconds')


def screenshot(page, path):
    page.screenshot(path=str(path), animations='disabled')


def capture_viewport(page, path):
    # Crop only the medical viewport. Full-browser screenshots vary with window size.
    box = viewport_box(page)
    page.screenshot(path=str(path), animations='disabled', clip=box)
    return box



def wait_for_demo(page, label):
    log(label)
    page.wait_for_timeout(CONFIG.get('demo_pause_ms', 1500))


def main():
    tests = load_test_cases(ROOT / CONFIG.get('test_cases_csv', 'test_cases.csv'))
    results = []
    plan = {'source': 'not generated', 'tests': []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=CONFIG.get('slow_mo_ms', 250))
        page = browser.new_page(viewport=CONFIG['viewport'], device_scale_factor=1)
        try:
            log('Opening OHIF study list')
            page.goto(CONFIG['url'], wait_until='domcontentloaded', timeout=60000)
            study = page.get_by_text(CONFIG['study'], exact=True)
            study.first.wait_for(state='visible', timeout=60000)
            count = study.count()
            log(f'Found {count} matching study labels')
            if count != 1:
                raise RuntimeError(f'Expected one study label; found {count}. Verify the study list.')
            study.dblclick(timeout=15000)
            log('Study selected; waiting for image canvas')
            page.wait_for_url('**/viewer?**', timeout=60000)
            deadline = time.monotonic() + 75
            while time.monotonic() < deadline:
                dismiss_tutorial(page)
                evidence = discover(page)
                if any(c['visible'] for c in evidence['canvas']):
                    break
                page.wait_for_timeout(1500)
            else:
                raise RuntimeError('No visible DICOM canvas was detected')
            # Allow asynchronous DICOM pixel rendering to settle before capturing baseline.
            page.wait_for_timeout(4000)
            dismiss_tutorial(page)
            evidence = discover(page)
            (OUTPUT / 'dom_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
            log('Agent 1: generating validated test definitions')
            plan = TestGenerationAgent().generate(evidence, CONFIG, OUTPUT, tests)
            (OUTPUT / 'generated_tests.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
            log(f'Plan source: {plan["source"]}')
            viewer_url = page.url
            verifier = VisualVerificationAgent()
            # Each baseline is created by a CSV row and reused by later rows.
            # Never silently overwrite an existing baseline_id within one run.
            created_baselines = set()
            for test in plan['tests']:
                tid, action = test['id'], test['action']
                wait_for_demo(page, f'Executing {tid}: {test["name"]}')
                page.set_viewport_size(test['viewport'])
                log(f'Viewport: {test["viewport"]}; mode: {test["comparison_mode"]}')
                actual = OUTPUT / f'{tid}_actual.png'
                diff = OUTPUT / f'{tid}_difference.png'
                expected = OUTPUT / f'baseline_{test["baseline_id"]}.png'
                if test['baseline_mode'] == 'create':
                    if test['baseline_id'] in created_baselines:
                        raise ValueError(f'Duplicate baseline creation: {test["baseline_id"]}')
                    if action == 'capture_baseline':
                        # TC01: no drawing, just capture and copy the baseline.
                        capture_viewport(page, expected)
                        actual.write_bytes(expected.read_bytes())
                    elif action == 'draw_measurement':
                        reload_clean_viewer(page, viewer_url)
                        geometry = draw_length(page, test['start'], test['end'], OUTPUT)
                        capture_viewport(page, expected)
                        (OUTPUT / f'{tid}_baseline_coordinates.json').write_text(
                            json.dumps(geometry, indent=2), encoding='utf-8')
                        # Real second drawing for the actual, not a copied image.
                        reload_clean_viewer(page, viewer_url)
                        geometry = draw_length(page, test['start'], test['end'], OUTPUT)
                        capture_viewport(page, actual)
                        (OUTPUT / f'{tid}_coordinates.json').write_text(
                            json.dumps(geometry, indent=2), encoding='utf-8')
                    created_baselines.add(test['baseline_id'])
                else:
                    if test['baseline_id'] not in created_baselines or not expected.exists():
                        raise RuntimeError(f'Missing baseline for {tid}: {test["baseline_id"]}')
                    if action != 'draw_measurement':
                        raise ValueError(f'Unsupported reuse action: {action}')
                    reload_clean_viewer(page, viewer_url)
                    geometry = draw_length(page, test['start'], test['end'], OUTPUT)
                    screenshot(page, actual)
                    (OUTPUT / f'{tid}_coordinates.json').write_text(
                        json.dumps(geometry, indent=2), encoding='utf-8')
                # Compare normalized viewport imagery and the requested annotation geometry.
                # Metadata checks are distinct from rendered-pixel checks.
                baseline_test = next(t for t in plan['tests'] if t['baseline_id'] == test['baseline_id'] and t['baseline_mode'] == 'create')
                check = verifier.compare(
                    expected, actual, diff, CONFIG['difference_threshold'],
                    mode=test['comparison_mode'],
                    baseline_geometry=baseline_test if action == 'draw_measurement' else None,
                    actual_geometry=test if action == 'draw_measurement' else None,
                    roi_threshold=CONFIG.get('roi_difference_threshold', 0.012),
                    coordinate_tolerance=CONFIG.get('coordinate_tolerance', 0.008))
                analysis = verifier.analyze_with_ollama(check, test, CONFIG, OUTPUT)
                results.append({'ai_analysis': analysis, 'id': tid, 'name': test['name'],
                                'status': check['status'], 'expected_status': test['expected_status'],
                                'matches_expected': check['status'] == test['expected_status'],
                                'reason': check['reason'], 'comparison_mode': test['comparison_mode'],
                                'viewport': test['viewport'], 'comparison_metrics': check, 'expected': expected.name,
                                'actual': actual.name, 'diff': diff.name})
                log(f'{tid} -> {check["status"]} (expected {test["expected_status"]}): {check["reason"]}')
        except Exception as exc:
            log(f'Execution error: {exc}')
            results.append({'id':'SETUP', 'name':'Open study and discover viewport', 'status':'FAIL', 'reason':str(exc)})
            try:
                page.screenshot(path=str(OUTPUT / 'setup_error.png'))
            except Exception:
                pass
        finally:
            if CONFIG.get('pause_before_close', True):
                try:
                    input('Review headed browser. Press Enter to close Chromium...')
                except (EOFError, KeyboardInterrupt):
                    pass
            browser.close()
    reporter = ReportingAgent()
    ai_report = reporter.analyze_with_ollama(results, plan, CONFIG, OUTPUT)
    summary = reporter.write(results, plan, OUTPUT, ai_report)
    log(f'Report: {OUTPUT / "report.html"}')
    log(f'Summary: {summary}')

if __name__ == '__main__':
    main()
