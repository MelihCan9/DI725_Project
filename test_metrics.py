import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from data_loader import RISCDataset
from tqdm import tqdm
import os
import json
import argparse

def validate(model, val_loader, processor, device, max_new_tokens=64, save_examples_path=None):
    model.eval()
    all_refs = []
    all_hyps = []
    examples = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch["image"].to(device)
            batch_captions = batch["captions"]
            for i in range(len(images)):
                captions = batch_captions[i]
                if not captions:
                    continue
                all_refs.extend(captions)
                inputs = processor(
                    images=images[i:i+1],
                    return_tensors="pt",
                    do_rescale=False
                ).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    length_penalty=0.65
                )
                generated_caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                generated_caption = generated_caption.replace("<image>", "").strip()
                all_hyps.extend([generated_caption] * len(captions))
                if len(examples) < 5:
                    examples.append({
                        "gt_captions": captions,
                        "generated": generated_caption
                    })
    if save_examples_path:
        with open(save_examples_path, "w") as f:
            json.dump(examples, f, indent=2)
    
    # Calculate metrics
    from nltk.translate.bleu_score import corpus_bleu
    from rouge_score import rouge_scorer
    from pycocoevalcap.cider.cider import Cider
    
    references_for_bleu = [[ref.split()] for ref in all_refs]
    hypotheses_for_bleu = [hyp.split() for hyp in all_hyps]
    bleu_score = corpus_bleu(references_for_bleu, hypotheses_for_bleu)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, hyp in zip(all_refs, all_hyps):
        scores = scorer.score(ref, hyp)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    for key in rouge_scores:
        rouge_scores[key] /= len(all_refs)
    
    cider_scorer = Cider()
    refs_for_cider = {i: [ref] for i, ref in enumerate(all_refs)}
    hyps_for_cider = {i: [hyp] for i, hyp in enumerate(all_hyps)}
    cider_score, _ = cider_scorer.compute_score(refs_for_cider, hyps_for_cider)
    
    metrics = {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'cider': cider_score
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save test results")
    args = parser.parse_args()

    # Constants
    MODEL_NAME = "google/paligemma-3b-pt-224"
    CSV_PATH = "/content/drive/MyDrive/DI/RISCM/captions.csv"
    IMAGES_DIR = "/content/drive/MyDrive/DI/RISCM/resized"
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    MAX_NEW_TOKENS = 64
    DEBUG_SAMPLE = 16  # Small subset for quick testing

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    missing, unexpected = base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model = base_model
    model.eval()

    # Prepare small validation dataset
    val_dataset = RISCDataset(CSV_PATH, IMAGES_DIR, split="val", image_size=IMAGE_SIZE)
    val_dataset = Subset(val_dataset, range(DEBUG_SAMPLE))
    print(f"Using {DEBUG_SAMPLE} validation samples for testing")

    def custom_collate(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["captions"] for item in batch]
        return {
            "image": images,
            "captions": captions,
            "image_id": [item["image_id"] for item in batch],
            "source": [item["source"] for item in batch],
        }

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )

    # Run validation
    val_metrics = validate(
        model, 
        val_loader, 
        processor, 
        DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        save_examples_path=os.path.join(args.output_dir, "test_examples.json")
    )

    # Print and save results
    print("\nTest Metrics:")
    print(f"BLEU: {val_metrics['bleu']:.4f}")
    print(f"ROUGE-1: {val_metrics['rouge1']:.4f}")
    print(f"ROUGE-2: {val_metrics['rouge2']:.4f}")
    print(f"ROUGE-L: {val_metrics['rougeL']:.4f}")
    print(f"CIDEr: {val_metrics['cider']:.4f}")

    # Save metrics to file
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=4)

if __name__ == "__main__":
    main() 