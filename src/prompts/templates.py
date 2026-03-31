"""All prompt templates for TrafficLLM as string constants."""

# ─── pexam: Demonstration example (from source cell) ──────────────────────────
PEXAM_TEMPLATE = (
    "The hourly network traffic experienced {input_values} Mbps on {input_date}.\n"
    "Based on historical hourly traffic from {input_date}, what will the predicted hourly "
    "traffic be on {output_date}?\n"
    "The predicted hourly traffic will {output_values} Mbps on {output_date}."
)

# ─── pinput: Input prompt ──────────────────────────────────────────────────────
PINPUT_TEMPLATE = (
    "The hourly network traffic experienced {input_values} Mbps on {input_date}."
)

# ─── pques: Question prompt ────────────────────────────────────────────────────
PQUES_TEMPLATE = (
    "Based on the historical hourly traffic from {input_date}, what will the predicted hourly "
    "traffic be on {target_date}? "
    "Please provide your prediction as 24 hourly values separated by commas."
)

# ─── Feedback Component 1: Overall Performance (MAE) ──────────────────────────
PFEED_MAE_TEMPLATE = (
    "Ground truth traffic on {target_date}: [{ground_truth}]\n"
    "Predicted traffic: [{predicted}]\n"
    "Calculate MAE = (1/n) \u03a3|predicted - actual|\n"
    'Format: "The MAE is: X.XX"'
)

# ─── Feedback Component 2: Periodical Performance (Sine/Cosine Fitting) ───────
PFEED_PERIODIC_TEMPLATE = (
    "For the following ground truth and predicted traffic, fit sine/cosine functions:\n"
    "Ground truth: [{ground_truth}]\n"
    "Predicted: [{predicted}]\n"
    "Fit: f_act = a*sin(w*t+p) + b*cos(w2*t+p2) + c\n"
    "     f_pred = a*sin(w*t+p) + c\n"
    "where t = hour (0-23)"
)

# ─── Validation sub-loop: review prompt ───────────────────────────────────────
VALIDATION_REVIEW_TEMPLATE = (
    "The previous mathematical calculation results are {previous_answers}. "
    "Please review the previous answers and find potential mistakes."
)

# ─── Validation sub-loop: correction prompt ───────────────────────────────────
VALIDATION_CORRECT_TEMPLATE = (
    "Please correct the answers based on the identified mistakes."
)

# ─── prefine: Refinement instruction ──────────────────────────────────────────
PREFINE_TEMPLATE = (
    "OK, let's predict hourly traffic on {target_date} again using the above feedback. "
    "More accurate prediction method should be considered. "
    "The overall error should be decreased. "
    "The prediction should match the function of ground truth as closely as possible."
)

# ─── Assembled pfeed body ──────────────────────────────────────────────────────
PFEED_ASSEMBLED_TEMPLATE = (
    '\u2022 Overall performance: "MAE is {mae_value} \u2014 error should be decreased"\n'
    '\u2022 Periodical performance: "f_act = {f_act_params} | f_pred = {f_pred_params} \u2014 '
    'prediction should match ground truth function"\n'
    '\u2022 Format: "prediction is complete and matches format."\n'
    '\u2022 Method: "{method_summary}"'
)

# ─── Growing prompt entry for a single prediction in refinement history ────────
REFINEMENT_HISTORY_INPUT = "Traffic experienced [{input_values}] on {input_date}."
REFINEMENT_HISTORY_PRED = "Predicted traffic [{pred_values}] on {target_date}."

# ─── Retry clarification prompt ───────────────────────────────────────────────
RETRY_CLARIFICATION_TEMPLATE = (
    "Please provide your answer as exactly 24 comma-separated numeric values. "
    "For example: 1.2, 3.4, 5.6, ... (24 values total). "
    "Do not include any other text, just the 24 numbers separated by commas."
)
