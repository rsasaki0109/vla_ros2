# Task-level success reported by a VLM watching the camera (vla_success_detector_node).
std_msgs/Header header
string task_id
string instruction
string vlm_name
string model_id
# Latched: true once required_consecutive checks in a row said yes; resets on a new instruction.
bool success
# Raw result of the most recent check.
bool last_check_success
# P(yes) of the most recent check; -1 when the VLM exposes no logits.
float32 confidence
string answer_text
uint32 consecutive_success
uint32 checks
float32 latency_ms
string metadata_json
