TEXTENC_DELIVERED 2026-11-11T20:40 CST (owner exp-2038)
UPDATE: the earlier "no deliverable" limitation is RESOLVED for the U16 path.
Delivered: build/c1of2_hp/textenc_L128_c1of2.axmodel (381,721,897 B, sha256 80bed2d8…)
           build/c2of2_hp/textenc_L128_c2of2.axmodel (384,311,393 B, sha256 57a9db93…)
chain cos (pulsar2 run vs ONNX fp32) = 0.9996766 (real tokens); segment c1 cos = 0.999846
config: S8 weights + all-U16 activations incl. Pulsar2 fused op types; c2 calibrated on chunk1 hidden (calib/c2_hidden)
See TEXTENC_REPORT.md section "BONUS" for root cause, A/B, reproduce commands and the two pitfalls
(fused ops stayed U8 when only ONNX op_type names were listed; chunk2 must be calibrated on chunk1 hidden states).
