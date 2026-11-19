# Học PyTorch

Lộ trình 6 bước, đi từ nền tảng đến Transformer và fine-tuning LLM.
Bạn đã biết lý thuyết DL (Coursera) và đã tự viết NN bằng numpy (`Deeplearning/nn-from-scratch`), nên trọng tâm là **thành thạo công cụ**.

| Bước | Thư mục | Nội dung | Bài tập chính |
|---|---|---|---|
| 1 | [01-tensor-autograd](01-tensor-autograd/main.ipynb) | Tensor, autograd, broadcasting, device | Linear regression chỉ bằng autograd |
| 2 | [quickstart](quickstart/main.ipynb) → [02-training-loop](02-training-loop/main.ipynb) | `nn.Module`, `DataLoader`, training loop | Viết lại quickstart từ trí nhớ, rồi cố tình làm hỏng |
| 3 | [03-cnn-regularization](03-cnn-regularization/main.ipynb) | CNN, dropout/BN, scheduler, logging | Bảng hyperparameter → accuracy |
| 4 | [04-text-rnn](04-text-rnn/main.ipynb) | Embedding, RNN/LSTM | Port bài NLP `week2/` sang PyTorch |
| 5 | [05-transformer-from-scratch](05-transformer-from-scratch/main.ipynb) | Attention, Transformer | Mini-GPT (theo nanoGPT) |
| 6 | [06-huggingface-finetune](06-huggingface-finetune/main.ipynb) | HF, LoRA/QLoRA | Fine-tune model nhỏ |

## Quy tắc học
1. **Gõ lại, rồi tự biến đổi** — không chỉ chạy cell của tutorial.
2. **In `shape`/`dtype`/`device`** ở mỗi bước; phần lớn bug nằm ở đó.
3. **`torch.manual_seed()`** để kết quả tái lập được.
4. **Cố tình làm hỏng** (bỏ `zero_grad`, bỏ `eval()`...) để hiểu vì sao chúng cần thiết.
5. Cuối mỗi notebook, ghi mục *Tổng kết* (học được gì / lỗi hay gặp / chưa hiểu).

## Tiến độ
- [ ] 1 · [ ] 2 · [ ] 3 · [ ] 4 · [ ] 5 · [ ] 6
