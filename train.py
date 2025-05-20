
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import RISCDataset
import random
import wandb
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import nltk
import os
import json
from datetime import datetime
import argparse

nltk.download('punkt')

def calculate_metrics(references, hypotheses):
    references_for_bleu = [[ref.split()] for ref in references]
    hypotheses_for_bleu = [hyp.split() for hyp in hypotheses]
    bleu_score = corpus_bleu(references_for_bleu, hypotheses_for_bleu)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
    for key in rouge_scores:
        rouge_scores[key] /= len(references)
    cider_scorer = Cider()
    refs_for_cider = {i: [ref] for i, ref in enumerate(references)}
    hyps_for_cider = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    cider_score, _ = cider_scorer.compute_score(refs_for_cider, hyps_for_cider)
    return {
        'bleu': bleu_score,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'cider': cider_score
    }

def validate(model, val_loader, processor, device, max_new_tokens=64, save_examples_path=None, debug_batch_limit=None):
    model.eval()
    all_refs = []
    all_hyps = []
    examples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
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
            if debug_batch_limit is not None and batch_idx + 1 >= debug_batch_limit:
                print(f"DEBUG: Sadece {debug_batch_limit} batch ile validation tamamlandı.")
                break
    if save_examples_path:
        with open(save_examples_path, "w") as f:
            json.dump(examples, f, indent=2)
    metrics = calculate_metrics(all_refs, all_hyps)
    model.train()
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Sanity check için mini dataset ve tek batch train/val")
    args = parser.parse_args()

    # ==== Sabitler ve dosya yolları ====
    MODEL_NAME = "google/paligemma-3b-pt-224"
    CSV_PATH = "/content/drive/MyDrive/DI/RISCM/captions.csv"
    IMAGES_DIR = "/content/drive/MyDrive/DI/RISCM/resized"
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8 if not args.debug else 2
    EVAL_BATCH_SIZE = 8 if not args.debug else 2
    GRAD_ACCUM_STEPS = 1
    EPOCHS = 2 if not args.debug else 1
    WARMUP_RATIO = 0.05
    EARLY_STOPPING_PATIENCE = 2
    MAX_NEW_TOKENS = 64

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"phase2-bench-lora{args.lora_r}_alpha{args.lora_alpha}_lr{args.learning_rate}_{timestamp}"
    OUTPUT_DIR = f"outputs/{run_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = {
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": args.learning_rate,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "image_size": IMAGE_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup_ratio": WARMUP_RATIO,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "debug_mode": args.debug
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    wandb.init(
        project="paligemma-phase2-benchmark",
        name=run_name,
        config=config
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # === DEBUG MINI-DATASET ===
    train_dataset = RISCDataset(CSV_PATH, IMAGES_DIR, split="train", image_size=IMAGE_SIZE)
    val_dataset = RISCDataset(CSV_PATH, IMAGES_DIR, split="val", image_size=IMAGE_SIZE)
    if args.debug:
        DEBUG_SAMPLE = 8
        train_dataset = Subset(train_dataset, range(DEBUG_SAMPLE))
        val_dataset = Subset(val_dataset, range(DEBUG_SAMPLE))
        print(f"DEBUG: Küçük veri ile test modunda! {DEBUG_SAMPLE} train, {DEBUG_SAMPLE} val örnek.")

    def custom_collate(batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["captions"] for item in batch]
        return {
            "image": images,
            "captions": captions,
            "image_id": [item["image_id"] for item in batch],
            "source": [item["source"] for item in batch],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=custom_collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        eps=1e-7
    )

    MAX_GRAD_NORM = 1.0
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    IMAGE_TOKEN = processor.tokenizer.additional_special_tokens[0]

    def prepend_image_token_to_caption(caption, image_token):
        caption = str(caption).strip()
        if caption.startswith(image_token):
            return caption
        return f"{image_token} {caption}"

    best_cider = -np.inf
    patience = 0

    def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        torch.save(checkpoint, filename)

    # === DEBUG: Bir batch ile ileri-gidiş testi ===
    if args.debug:
        print("DEBUG: Model forward testi başlıyor")
        train_iter = iter(train_loader)
        test_batch = next(train_iter)
        images = test_batch["image"]
        captions = test_batch["captions"]
        print(f"Sample image shape: {images.shape}, Sample captions: {captions[0]}")

        inputs = processor(
            images=images,
            text=[f"<image> {random.choice(cap if cap else ['empty'])}" for cap in captions],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
            do_rescale=False,
            add_special_tokens=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        with torch.no_grad():
            out = model(**inputs, labels=labels)
        print("DEBUG: Model forward çalıştı, loss:", out.loss.item())

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            images = batch["image"]
            batch_captions = batch["captions"]
            batch_size = len(images)

            captions = []
            for i in range(batch_size):
                image_captions = batch_captions[i]
                if not image_captions:
                    captions.append(prepend_image_token_to_caption("empty", IMAGE_TOKEN))
                else:
                    selected_caption = random.choice(image_captions)
                    captions.append(prepend_image_token_to_caption(selected_caption, IMAGE_TOKEN))

            assert len(images) == len(captions), f"Batch image and caption count mismatch: {len(images)} vs {len(captions)}"

            inputs = processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
                do_rescale=False,
                add_special_tokens=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{running_loss / (step + 1):.4f}',
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

            if (step + 1) % 50 == 0:
                wandb.log({
                    "train/loss": running_loss / (step + 1),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + 1,
                    "train/step": step + 1
                })

            # === DEBUG: Sadece bir batch ilerlet
            if args.debug and step == 0:
                print("DEBUG: İlk batch işlendikten sonra eğitim kesiliyor (debug mode)")
                break

        epoch_loss = running_loss / (step + 1 if args.debug else len(train_loader))
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": epoch_loss
        })

        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.4f}")

        val_metrics = validate(
            model, val_loader, processor, DEVICE,
            max_new_tokens=MAX_NEW_TOKENS,
            save_examples_path=os.path.join(OUTPUT_DIR, f"val_examples_epoch{epoch+1}.json"),
            debug_batch_limit=1 if args.debug else None  # debug'da sadece 1 batch
        )
        wandb.log({
            "val/bleu": val_metrics['bleu'],
            "val/rouge1": val_metrics['rouge1'],
            "val/rouge2": val_metrics['rouge2'],
            "val/rougeL": val_metrics['rougeL'],
            "val/cider": val_metrics['cider']
        })
        print(f"Epoch {epoch+1} Validation Metrics:")
        print(f"BLEU: {val_metrics['bleu']:.4f}")
        print(f"ROUGE-1: {val_metrics['rouge1']:.4f}")
        print(f"ROUGE-2: {val_metrics['rouge2']:.4f}")
        print(f"ROUGE-L: {val_metrics['rougeL']:.4f}")
        print(f"CIDEr: {val_metrics['cider']:.4f}")

        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_metrics, checkpoint_path
        )
        if val_metrics['cider'] > best_cider:
            best_cider = val_metrics['cider']
            patience = 0
            best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, best_model_path
            )
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    main()
