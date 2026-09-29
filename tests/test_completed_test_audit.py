import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import audit_test_completion as audit
from scripts import test_evaluation as ev
from scripts import test_recovery as recovery
from test_test_recovery import fixture


class CompletionAuditTests(unittest.TestCase):
    def test_console_encodings(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'console.log'
            for encoding in ('utf-8','utf-8-sig','utf-16','utf-16-le'):
                p.write_text('TEST EVALUATION COMPLETE',encoding=encoding)
                self.assertEqual(audit.console_text(p),'TEST EVALUATION COMPLETE')

    def test_complete_fictional_audit_no_inference(self):
        with fixture() as (root,bundle):
            current,e,task,_,_,_,blind=bundle
            rows,_,_=recovery.verify_prefix(ev.OUTPUT,root,blind,task,e,18)
            with patch.object(recovery,'prepare',return_value=(bundle,current,rows)):
                ev.execute(root/'receipt.json',recovery_audit=root/'unused')
            out=ev.OUTPUT; patients=ev.read(out/'patients.json'); marker=ev.read(out/'complete.json')
            log=root/'log'; log.write_text('TEST EVALUATION COMPLETE')
            with patch.object(audit,'metadata',return_value=(current,e,task,root,blind,patients,marker,log)), \
                 patch.object(ev,'BASE',root),patch.object(ev,'score_patient',side_effect=AssertionError('No inference allowed')):
                target=root/'review-audit'; audit.audit(target); ev.verify_seal(target)
                result=ev.read(target/'audit_results.json')
                self.assertEqual(result['windows'],18); self.assertFalse(result['inference_executed'])
                with self.assertRaises(ValueError): audit.audit(target)
                with open(out/'patient-000000.npz','ab') as f: f.write(b'tampered')
                with self.assertRaises(ValueError): audit.audit(root/'bad-audit')


if __name__=='__main__': unittest.main()
