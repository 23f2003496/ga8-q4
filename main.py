from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, List
import decimal
import re

app = FastAPI()

# Constants for fixed order of interventions
INTERVENTIONS = ["prompt_only", "retrieval", "lora", "qlora"]

def is_finite_float_0_1(val):
    try:
        f = float(val)
        if not (0 <= f <= 1):
            return False
        return f == f and f not in (float('inf'), float('-inf'))
    except:
        return False

def is_nonneg_finite_float(val):
    try:
        f = float(val)
        return f == f and f >= 0 and f not in (float('inf'), float('-inf'))
    except:
        return False

def is_nonneg_int(val):
    try:
        i = int(val)
        return i >= 0
    except:
        return False

def sort_utf8_unique(lst):
    return sorted(set(lst), key=lambda x: x.encode('utf-8'))

def valid_hex_lowercase(s, length):
    return isinstance(s, str) and len(s) == length and re.fullmatch(r'[0-9a-f]{' + str(length) + r'}', s)

@app.post("/adapt")
async def adapt(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    operation = data.get("operation")
    if operation == "choose":
        policy = data.get("policy")
        candidates = data.get("candidates")

        # Validate minimal inputs
        if not (isinstance(policy, dict) and isinstance(candidates, list)):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        # Confirm exact one candidate per intervention and all four interventions present
        cand_map = {c.get("name"): c for c in candidates if isinstance(c, dict) and "name" in c}
        if any(interv not in cand_map or len([c for c in candidates if c.get("name") == interv]) != 1 for interv in INTERVENTIONS):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        totalCosts = {}
        reasonCodes = {}

        eligible = []
        for interv in INTERVENTIONS:
            cand = cand_map[interv]
            codes = []

            # Validate fields exist and type checks
            if not isinstance(cand.get("available"), bool):
                codes.append("INVALID_INPUT")
            if not (cand.get("quality") is not None and is_finite_float_0_1(cand.get("quality"))):
                codes.append("INVALID_INPUT")
            if not isinstance(cand.get("freshness"), bool):
                codes.append("INVALID_INPUT")
            if not (is_nonneg_finite_float(cand.get("latencyMs"))):
                codes.append("INVALID_INPUT")
            if not (is_nonneg_finite_float(cand.get("memoryMb"))):
                codes.append("INVALID_INPUT")
            if not (is_nonneg_int(cand.get("labeledExamples"))):
                codes.append("INVALID_INPUT")
            if not (is_nonneg_finite_float(cand.get("oneTimeCost"))):
                codes.append("INVALID_INPUT")
            if not (is_nonneg_finite_float(cand.get("recurringCost"))):
                codes.append("INVALID_INPUT")
            
            # Compute cost even if invalid, to be safe - round 12 decimals
            oneTimeCost = float(cand.get("oneTimeCost", 0))
            recurringCost = float(cand.get("recurringCost", 0))
            horizonRequests = policy.get("horizonRequests")
            if not is_nonneg_int(horizonRequests):
                return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
            total_cost = oneTimeCost + horizonRequests * recurringCost
            total_cost = round(total_cost, 12)
            totalCosts[interv] = total_cost

            # Gates
            if not cand.get("available", False):
                codes.append("UNAVAILABLE")
            if cand.get("quality", 0) < policy.get("minQuality", 1e9):
                codes.append("QUALITY_FLOOR")
            if policy.get("freshnessRequired", False) and not cand.get("freshness", False):
                codes.append("FRESHNESS_REQUIRED")
            if cand.get("latencyMs", 1e9) > policy.get("maxLatencyMs", 0):
                codes.append("LATENCY_LIMIT")
            if cand.get("memoryMb", 1e9) > policy.get("maxMemoryMb", 0):
                codes.append("MEMORY_LIMIT")
            if cand.get("labeledExamples", 1e9) > policy.get("maxLabeledExamples", 0):
                codes.append("DATA_LIMIT")
            if total_cost > policy.get("maxTotalCost", 0):
                codes.append("COST_LIMIT")

            reasonCodes[interv] = sort_utf8_unique(codes)
            if not codes:
                eligible.append(cand)

        # Sort eligible in the published priority order (INTERVENTIONS)
        eligible_sorted = [c for i in INTERVENTIONS for c in eligible if c["name"] == i]

        selected = eligible_sorted[0] if eligible_sorted else None
        selected_name = selected["name"] if selected else None

        # Prepare output reasonCodes for all interventions
        # Ensure keys exist for all
        for interv in INTERVENTIONS:
            if interv not in reasonCodes:
                reasonCodes[interv] = []

        return {
            "selected": selected_name,
            "eligible": [c["name"] for c in eligible_sorted],
            "totalCosts": {k: totalCosts.get(k, 0) for k in INTERVENTIONS},
            "reasonCodes": {k: reasonCodes.get(k, []) for k in INTERVENTIONS}
        }

    elif operation == "repair":
        tokens = data.get("tokens")
        templateApplications = data.get("templateApplications")
        parameters = data.get("parameters")
        allowedTargets = data.get("allowedTargets")
        inferenceMode = data.get("inferenceMode")
        trainRowIds = data.get("trainRowIds")
        evalRowIds = data.get("evalRowIds")
        dropoutActiveDuringEval = data.get("dropoutActiveDuringEval")
        artifactFiles = data.get("artifactFiles")
        baseRevision = data.get("baseRevision")
        datasetDigest = data.get("datasetDigest")
        codeDigest = data.get("codeDigest")
        configDigest = data.get("configDigest")
        expectedDigests = data.get("expectedDigests")
        microBatch = data.get("microBatch")
        gradientAccumulation = data.get("gradientAccumulation")
        replicas = data.get("replicas")
        expectedEffectiveBatch = data.get("expectedEffectiveBatch")
        checkpoint = data.get("checkpoint")
        uninterruptedWeights = data.get("uninterruptedWeights")
        resumedWeights = data.get("resumedWeights")
        resumeTolerance = data.get("resumeTolerance")

        reasonCodes = []
        labels = []

        # Check tokens validity and label generation
        valid_tokens = True
        if not isinstance(tokens, list) or len(tokens) == 0:
            valid_tokens = False
        else:
            ids_seen = set()
            for t in tokens:
                if (not isinstance(t, dict) or 
                    "id" not in t or "role" not in t or "padding" not in t or "text" not in t):
                    valid_tokens = False
                    break
                if not (is_nonneg_int(t["id"]) and t["role"] in ("system", "user", "assistant") and
                        isinstance(t["padding"], bool) and isinstance(t["text"], str)):
                    valid_tokens = False
                    break
                if t["id"] in ids_seen:
                    valid_tokens = False
                    break
                ids_seen.add(t["id"])
            if valid_tokens:
                for t in tokens:
                    if t["role"] == "assistant" and not t["padding"]:
                        labels.append(t["id"])
                    else:
                        labels.append(-100)
            else:
                labels = [-100] * len(tokens)

        # Validate exactly one template application
        templatePass = templateApplications == 1

        # Parameter validation
        trainableParams = []
        trainableCount = 0
        peftConfigPass = True
        param_names = set()
        allowed_targets_set = set(allowedTargets) if isinstance(allowedTargets, list) else set()

        if (not isinstance(parameters, list) or len(parameters) == 0 or
            not isinstance(allowedTargets, list) or len(allowedTargets) == 0):
            peftConfigPass = False
        else:
            for p in parameters:
                if (not isinstance(p, dict) or "name" not in p or "target" not in p or "numel" not in p):
                    peftConfigPass = False
                    break
                if not (isinstance(p["name"], str) and isinstance(p["target"], str) and is_nonneg_int(p["numel"]) and p["numel"] > 0):
                    peftConfigPass = False
                    break
                if p["name"] in param_names:
                    peftConfigPass = False
                    break
                param_names.add(p["name"])
                if p["target"] in allowed_targets_set and (p["name"].endswith(".lora_A.weight") or p["name"].endswith(".lora_B.weight")):
                    trainableParams.append(p)
                    trainableCount += p["numel"]
            if len(set(allowedTargets)) != len(allowedTargets):
                peftConfigPass = False

        # inferenceMode and dropoutActiveDuringEval checks
        if inferenceMode != False:
            reasonCodes.append("INFERENCE_MODE")
        if dropoutActiveDuringEval != False:
            reasonCodes.append("EVAL_DROPOUT_ACTIVE")

        # Check trainRowIds and evalRowIds non-empty unique sets, disjoint sets
        if (not isinstance(trainRowIds, list) or not isinstance(evalRowIds, list) or
            len(trainRowIds) == 0 or len(evalRowIds) == 0):
            reasonCodes.append("INVALID_INPUT")
        else:
            set_train = set(trainRowIds)
            set_eval = set(evalRowIds)
            if len(set_train) != len(trainRowIds) or len(set_eval) != len(evalRowIds) or len(set_train.intersection(set_eval)) != 0:
                reasonCodes.append("INVALID_INPUT")

        # artifactFiles must exactly be ["adapter_config.json", "adapter_model.safetensors"] sorted by UTF8 bytes
        if (not isinstance(artifactFiles, list) or 
            sorted(artifactFiles) != ["adapter_config.json", "adapter_model.safetensors"]):
            reasonCodes.append("ADAPTER_FILE_SET")

        # baseRevision must be 40-lowercase-hex, digests 64-lowercase-hex non-empty strings
        if not valid_hex_lowercase(baseRevision, 40):
            reasonCodes.append("MUTABLE_BASE_REVISION")
        if not (valid_hex_lowercase(datasetDigest, 64) and valid_hex_lowercase(codeDigest, 64) and valid_hex_lowercase(configDigest, 64)):
            reasonCodes.append("LINEAGE_MISMATCH")

        # expectedEffectiveBatch == microBatch * gradientAccumulation * replicas positive safe integers
        if (not is_nonneg_int(microBatch) or microBatch < 1 or
            not is_nonneg_int(gradientAccumulation) or gradientAccumulation < 1 or
            not is_nonneg_int(replicas) or replicas < 1 or
            not is_nonneg_int(expectedEffectiveBatch) or expectedEffectiveBatch < 1):
            reasonCodes.append("EFFECTIVE_BATCH_MISMATCH")
        else:
            if microBatch * gradientAccumulation * replicas != expectedEffectiveBatch:
                reasonCodes.append("EFFECTIVE_BATCH_MISMATCH")

        # checkpoint must own model, optimizer, scheduler, step, rng, dataPosition
        checkpoint_keys = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}
        if not isinstance(checkpoint, dict) or set(checkpoint.keys()) != checkpoint_keys:
            reasonCodes.append("INCOMPLETE_CHECKPOINT")

        # Resume arrays checks
        def check_array(arr):
            if not isinstance(arr, list) or len(arr) == 0:
                return False
            for x in arr:
                if not isinstance(x, (float,int)):
                    return False
            return True
        
        if (not check_array(uninterruptedWeights) or
            not check_array(resumedWeights) or
            len(uninterruptedWeights) != len(resumedWeights)):
            reasonCodes.append("RESUME_DIVERGENCE")
        
        if not (is_nonneg_finite_float(resumeTolerance)):
            reasonCodes.append("RESUME_DIVERGENCE")
        else:
            # Check absolute differences
            for a, b in zip(uninterruptedWeights, resumedWeights):
                if abs(a - b) > resumeTolerance:
                    reasonCodes.append("RESUME_DIVERGENCE")
                    break

        # Compose response
        reasonCodesSorted = sort_utf8_unique(reasonCodes)

        return {
            "labels": labels,
            "templatePass": templatePass,
            "trainableParams": sorted([p["name"] for p in trainableParams], key=lambda x: x.encode("utf-8")),
            "trainableCount": trainableCount,
            "peftConfigPass": peftConfigPass,
            "adapterFiles": sorted(artifactFiles) if isinstance(artifactFiles, list) else [],
            "checkpointComplete": "INCOMPLETE_CHECKPOINT" not in reasonCodesSorted,
            "lineagePass": "LINEAGE_MISMATCH" not in reasonCodesSorted and "MUTABLE_BASE_REVISION" not in reasonCodesSorted,
            "evalIsolated": "EVAL_LEAKAGE" not in reasonCodesSorted,
            "evaluationDeterministic": True,
            "resumePass": "RESUME_DIVERGENCE" not in reasonCodesSorted,
            "reasonCodes": reasonCodesSorted
        }

    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

