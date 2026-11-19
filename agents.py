import json
from pathlib import Path
import requests
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

ROOT = Path(__file__).resolve().parent

def ask_ollama(config, role, payload, output_dir, max_tokens=240):
    """One auditable LLM invocation; never silently misrepresent fallback as AI."""
    output_dir = Path(output_dir)
    safe_role = role.lower().replace(' ', '_')
    request = {
        'model': config['ollama_model'], 'stream': False, 'format': 'json',
        'messages': [
            {'role': 'system', 'content': role + '. Respond with one short valid JSON object only.'},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, default=str)[:4000]}
        ],
        'options': {'temperature': 0, 'num_predict': max_tokens}
    }
    (output_dir / f'{safe_role}_input.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    try:
        response = requests.post(config['ollama_url'].rstrip('/') + '/api/chat', json=request,
                                 timeout=config.get('ollama_timeout_seconds', 90))
        response.raise_for_status()
        content = response.json()['message']['content']
        result = json.loads(content)
        (output_dir / f'{safe_role}_output.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return {'called': True, 'success': True, 'result': result}
    except Exception as exc:
        error = {'called': True, 'success': False, 'error': str(exc)}
        (output_dir / f'{safe_role}_output.json').write_text(json.dumps(error, indent=2), encoding='utf-8')
        return error

class TestGenerationAgent:
    """Ollama reviews real UI evidence and CSV scenarios; CSV remains authoritative."""
    def generate(self, evidence, config, output_dir, tests):
        compact = {'resolution': evidence.get('resolution'), 'canvas': evidence.get('canvas', [])[:3],
                   'controls': evidence.get('controls', [])[:15]}
        llm = ask_ollama(config, 'Agent 1 CSV test planning',
            {'task': 'Review these user-supplied CSV test cases against discovered browser evidence. '
                     'Return JSON with keys summary, risks, recommendations. Do not invent or change test actions.',
             'tests': tests, 'evidence': compact}, output_dir)
        source = 'CSV definitions (Ollama review succeeded)' if llm['success'] else 'CSV definitions (Ollama review failed)'
        return {'source': source, 'ollama': llm, 'evidence': evidence, 'tests': tests}

class VisualVerificationAgent:
    def compare(self, expected_path, actual_path, diff_path, threshold,
                mode='same_resolution', baseline_geometry=None, actual_geometry=None,
                roi_threshold=0.012, coordinate_tolerance=0.008):
        """Compare viewport crops. Resolution-independent mode normalizes canvas sizes.

        The geometry guard compares CSV-requested normalized endpoints; it is NOT
        proof that OHIF rendered the annotation. Pixel comparison remains required.
        """
        from PIL import ImageDraw
        with Image.open(expected_path) as exp_img, Image.open(actual_path) as act_img:
            expected = exp_img.convert('RGB')
            actual = act_img.convert('RGB')
            if mode == 'same_resolution' and expected.size != actual.size:
                return {'status':'FAIL', 'reason':'Viewport image dimensions differ in strict comparison mode',
                        'difference_ratio':1.0, 'mode':mode}
            if mode == 'resolution_independent':
                # Normalization handles scaling, not changes in anatomical framing.
                actual = actual.resize(expected.size, Image.Resampling.LANCZOS)
            a = np.asarray(expected, dtype=np.int16)
            b = np.asarray(actual, dtype=np.int16)
            pixel_mask = np.any(np.abs(a - b) > 20, axis=2)
            ratio = float(pixel_mask.mean())
            difference = ImageChops.difference(expected, actual)
            ImageEnhance.Contrast(difference).enhance(4).save(diff_path)

            metrics = {'mode':mode, 'difference_ratio':round(ratio, 6),
                       'expected_image_size':list(expected.size), 'actual_image_size':list(act_img.size)}
            # A local ROI is more sensitive to thin measurements than the whole image.
            if baseline_geometry and actual_geometry:
                points = [baseline_geometry['start'], baseline_geometry['end'],
                          actual_geometry['start'], actual_geometry['end']]
                w,h = expected.size
                pad = 0.035
                x0 = max(0, int((min(pt[0] for pt in points)-pad)*w))
                y0 = max(0, int((min(pt[1] for pt in points)-pad)*h))
                x1 = min(w, int((max(pt[0] for pt in points)+pad)*w))
                y1 = min(h, int((max(pt[1] for pt in points)+pad)*h))
                roi_ratio = float(pixel_mask[y0:y1, x0:x1].mean()) if x1>x0 and y1>y0 else 1.0
                deviation = max(abs(u-v) for p,q in zip(
                    [baseline_geometry['start'],baseline_geometry['end']],
                    [actual_geometry['start'],actual_geometry['end']]) for u,v in zip(p,q))
                metrics.update(roi_difference_ratio=round(roi_ratio,6),
                               requested_endpoint_deviation=round(deviation,6),
                               roi_bounds=[x0,y0,x1,y1])
                # A changed test input must not pass just because its line occupies few pixels.
                geometry_ok = deviation <= coordinate_tolerance
                visual_ok = roi_ratio <= roi_threshold
                status = 'PASS' if geometry_ok and visual_ok else 'FAIL'
                reason = (f'ROI {roi_ratio:.2%} (allowed {roi_threshold:.2%}); '
                          f'requested endpoint delta {deviation:.3f} (allowed {coordinate_tolerance:.3f}); '
                          f'full viewport difference {ratio:.2%}.')
            else:
                status = 'PASS' if ratio <= threshold else 'FAIL'
                reason = f'Full viewport difference {ratio:.2%} (allowed {threshold:.2%})'
            return {'status':status, 'reason':reason, **metrics}

    def analyze_with_ollama(self, comparison, test, config, output_dir):
        """LLM interprets measured visual differences; numeric thresholds own PASS/FAIL."""
        payload = {'task': 'Explain the visual result briefly. Return JSON with keys assessment, likely_cause, recommendation. '
                           'Do not claim to have viewed pixels or diagnose clinical image quality.',
                   'test': test, 'comparison': comparison,
                   'intentional_failure': test.get('expected_status') == 'FAIL'}
        return ask_ollama(config, 'Agent 2 visual analysis ' + test['id'], payload, output_dir)

class ReportingAgent:
    def analyze_with_ollama(self, results, plan, config, output_dir):
        payload = {'task': 'Provide a short QA executive assessment as JSON with keys summary, risks, next_steps. '
                           'Respect observed PASS/FAIL and label intentional simulated failure.',
                   'plan_source': plan.get('source'),
                   'results': [{k: r.get(k) for k in ('id', 'name', 'status', 'reason', 'ai_analysis')} for r in results]}
        return ask_ollama(config, 'Agent 3 reporting', payload, output_dir, max_tokens=320)

    def write(self, results, plan, output_dir, ai_report=None):
        import html
        output_dir = Path(output_dir)
        summary = {'total': len(results), 'passed': sum(x['status'] == 'PASS' for x in results),
                   'failed': sum(x['status'] == 'FAIL' for x in results)}
        (output_dir / 'report.json').write_text(json.dumps({'summary': summary, 'plan': plan, 'results': results, 'agent3_ollama': ai_report}, indent=2), encoding='utf-8')
        rows = []
        for r in results:
            evidence = ''.join(f'<a href="{html.escape(str(r[k]))}">{k}</a> ' for k in ('expected', 'actual', 'diff') if r.get(k))
            rows.append(f'<tr><td>{html.escape(r["id"])}</td><td>{html.escape(r["name"])}</td>'
                        f'<td class="{r["status"].lower()}">{r["status"]}</td>'
                        f'<td>{html.escape(r.get("reason", ""))}</td><td>{evidence}</td></tr>')
        page = f'''<!doctype html><html><head><meta charset="utf-8"><title>DICOM Visual Test Report</title>
<style>body{{font-family:Arial,sans-serif;background:#111827;color:#f9fafb;margin:35px}}h1{{color:#93c5fd}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #374151;padding:13px;text-align:left}}.pass{{color:#4ade80;font-weight:bold}}.fail{{color:#f87171;font-weight:bold}}a{{color:#93c5fd}}.summary{{padding:15px;background:#1f2937;border-radius:8px;margin:20px 0}}</style></head>
<body><h1>DICOM Visual Verification — Multi-Agent Demo</h1><div class="summary">Total: {summary['total']} &nbsp; Passed: {summary['passed']} &nbsp; Failed: {summary['failed']}<p>Plan source: {html.escape(plan['source'])}</p><p>Agent 3 Ollama: {html.escape(json.dumps(ai_report, ensure_ascii=False)[:1400] if ai_report else 'not run')}</p></div><table><thead><tr><th>ID</th><th>Test</th><th>Status</th><th>Visual comparison</th><th>Evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>Rows marked expected_status=FAIL are intentional negative test scenarios, not necessarily application defects. Baselines were created during this run, not independently approved references.</p></body></html>'''
        (output_dir / 'report.html').write_text(page, encoding='utf-8')
        return summary
