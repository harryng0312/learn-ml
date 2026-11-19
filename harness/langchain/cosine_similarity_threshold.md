# Ngưỡng Cosine Similarity

## Range và ý nghĩa (Sentence Transformers)

| Score | Ý nghĩa thực tế |
|-------|----------------|
| `> 0.90` | Gần như giống hệt hoặc diễn đạt cùng 1 ý |
| `0.75 – 0.90` | Cùng chủ đề, rất liên quan |
| `0.50 – 0.75` | Có liên quan, cùng lĩnh vực |
| `0.25 – 0.50` | Liên quan lờ mờ |
| `< 0.25` | Không liên quan |

## Ngưỡng thực tế hay dùng

| Bài toán | Ngưỡng |
|----------|--------|
| Semantic search / RAG | Lấy top-k, không dùng threshold cứng. Nếu cần filter thì `> 0.5` |
| Duplicate detection | `> 0.85` thì coi là trùng |
| Clustering | `> 0.6` là cùng nhóm |

## Lưu ý

Score phụ thuộc vào **model** — cùng 2 câu nhưng khác model cho score khác nhau.
Không nên hardcode ngưỡng, thay vào đó calibrate trên dữ liệu thực:

```python
# Lấy score của vài cặp "chắc chắn liên quan" và "chắc chắn không liên quan"
# → chọn ngưỡng nằm giữa 2 nhóm đó
```
