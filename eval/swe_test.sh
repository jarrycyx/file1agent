export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export ALL_PROXY="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="http://127.0.0.1:7890"
export no_proxy="localhost,127.0.0.1,.local"


export OPENAI_API_KEY="0"
export OPENAI_BASE_URL="http://127.0.0.1:8077/v1"
# export OPENAI_API_KEY="sk-uewsuzpc332yhydw"
# export OPENAI_BASE_URL="https://cloud.infini-ai.com/maas/v1/"
# export MSWEA_COST_TRACKING='ignore_errors'
export LITELLM_MODEL_REGISTRY_PATH="eval/models.json"

rm -rf outputs/swebench

# python eval/mini-swe-agent/src/minisweagent/run/extra/swebench.py\
#     --model openai/glm-4.5-air \
#     --subset verified \
#     --split test \
#     --workers 4 \
#     --output outputs/swebench/

python eval/swebench_file1.py\
    --model openai/glm-4.5-air \
    --subset verified \
    --split test \
    --workers 4 \
    --output outputs/swebench/

cd eval/SWE-bench ; \
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path ../../outputs/swebench/preds.json \
    --max_workers 4 \
    --run_id eval-glm-4.5-air ; \
cd ../../