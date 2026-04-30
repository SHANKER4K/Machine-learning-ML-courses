# نموذج توليد النصوص - Text Generation with PyTorch

## 📖 نبذة

برنامج كامل لتدريب نموذج توليد نصوص عربية باستخدام PyTorch.
يدعم نموذجي LSTM و Transformer.

---

## 🚀 البدء السريع

### 1. التثبيت

```bash
pip install torch numpy
```

### 2. تجهيز البيانات

ضع ملف بيانات باسم `islamic_texts.txt` يحتوي على نصوص عربية (سطر واحد لكل نص):

```
الحمد لله رب العالمين
قال الله تعالى...
```

### 3. التشغيل

```bash
python text_generation_complete.py
```

---

## 📚 الهيكل

```
text_generation_complete.py
├── TextPreprocessor       # تنظيف البيانات
├── SimpleArabicTokenizer  # تحويل النصوص لأرقام
├── TextDataset           # مجموعة البيانات
├── LSTMTextGenerator     # نموذج LSTM
├── TransformerTextGenerator  # نموذج Transformer
├── TextGenerationTrainer # تدريب النموذج
├── TextGenerator         # توليد النصوص
└── main()               # البرنامج الرئيسي
```

---

## ⚙️ التعديلات

### تغيير حجم النموذج

في دالة `main()`:

```python
# اختر بين LSTM و Transformer
MODEL_TYPE = "transformer"  # أو "lstm"
```

### تغيير عدد الـ Epochs

```python
NUM_EPOCHS = 10  # اجعلها أكبر للدقة أفضل (لكن أبطأ)
```

### تغيير حجم Batch

```python
BATCH_SIZE = 32  # اجعلها أصغر إذا كان عندك ذاكرة أقل
```

### تغيير عدد المفردات

```python
VOCAB_SIZE = 10000  # عدد الكلمات الفريدة
```

---

## 📊 النتائج

بعد التدريب ستجد:
- `best_model.pth` — أفضل نموذج محفوظ
- `text_generator_transformer.pth` — النموذج النهائي
- `tokenizer.json` — قاموس الكلمات

---

## 🔧 استخدام النموذج المحفوظ

```python
from text_generation_complete import *
import torch

# تحميل Tokenizer
tokenizer = SimpleArabicTokenizer()
tokenizer.load("tokenizer.json")

# تحميل النموذج
model = TransformerTextGenerator(vocab_size=10000)
model.load_state_dict(torch.load("text_generator_transformer.pth"))

# توليد نص
device = torch.device("cpu")
generator = TextGenerator(model, tokenizer, device)

text = generator.generate("الحمد لله", max_length=100)
print(text)
```

---

## 💡 نصائح

1. **البيانات أهم من النموذج** — احصل على كمية بيانات كبيرة
2. **استخدم GPU** — التدريب أسرع بـ 100x
3. **جرب Transformer أولاً** — أفضل من LSTM في معظم الحالات
4. **Temperature عند التوليد** — 
   - 0 = نتائج متطابقة (محافظ)
   - 0.7 = متوازن (الأفضل)
   - 1+ = عشوائي أكثر (إبداعي)
5. **Top-k Sampling** — يقلل النتائج السيئة

---

## 📈 تحسينات مستقبلية

- [ ] إضافة Attention Visualization
- [ ] استخدام Pre-trained Models (AraBERT, ARAGPT2)
- [ ] Fine-tuning على نماذج موجودة
- [ ] إضافة Beam Search
- [ ] تقييم BLEU Score و Perplexity

---

## 🐛 استكشاف الأخطاء

### "CUDA out of memory"
→ اجعل BATCH_SIZE أصغر أو استخدم CPU

### "File not found"
→ تأكد من وجود `islamic_texts.txt` في نفس المجلد

### Loss لا ينخفض
→ قلل Learning Rate أو اجعل البيانات أنظف

---

## 📞 أسئلة؟

للمساعدة: سيدي إسماعيل 💚

---

**تاريخ الإنشاء:** 22 أبريل 2026
**النسخة:** 1.0
