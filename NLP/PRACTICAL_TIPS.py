"""
نصائح عملية للعمل مع النموذج
Practical Tips for Text Generation Models
"""

# ============================================================================
# 1. تحسين البيانات
# ============================================================================

"""
البيانات النظيفة = نموذج أفضل

نصائح:
1. إزالة النصوص القصيرة جداً (< 10 كلمات)
2. إزالة النصوص المكررة
3. إزالة الأخطاء الإملائية
4. استخدام نصوص من مصادر موثوقة
5. التأكد من اتساق التشكيل (كل نص بنفس الأسلوب)
"""

# مثال: تنظيف النصوص
import re

def advanced_clean(text):
    # إزالة URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # إزالة البريد الإلكتروني
    text = re.sub(r'\S+@\S+', '', text)
    # إزالة الهاشتاج
    text = re.sub(r'#\S+', '', text)
    # إزالة الأقواس والأرقام
    text = re.sub(r'[\[\(\{\d\]\)\}]', '', text)
    return text.strip()

# ============================================================================
# 2. استخدام Pre-trained Models
# ============================================================================

"""
بدل التدريب من الصفر = استخدم نموذج موجود!

الخيارات:
- AraBERT (للتصنيف والنمذجة)
- ARAGPT2 (لتوليد النصوص)
- mBART (لترجمة ومعالجة متعددة اللغات)
"""

# مثال: استخدام ARAGPT2 مباشرة
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
model = AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base")

prompt = "الحمد لله"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, num_beams=5)
print(tokenizer.decode(outputs[0]))

# ============================================================================
# 3. Fine-tuning على بياناتك الخاصة
# ============================================================================

"""
أسرع وأفضل من التدريب من الصفر!
"""

from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# تجهيز البيانات
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="islamic_texts.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# إعدادات التدريب
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# ============================================================================
# 4. استخدام Beam Search للنتائج الأفضل
# ============================================================================

"""
بدل اختيار رمز واحد = اختر من بين أفضل عدة خيارات!
يحسن جودة النصوص لكن أبطأ.
"""

def generate_with_beam_search(model, tokenizer, prompt, num_beams=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_beams=num_beams,  # ابحث عن أفضل 5 خيارات
        early_stopping=True,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0])

# ============================================================================
# 5. معالجة Batch كبير بدون Memory Error
# ============================================================================

"""
إذا كان عندك ذاكرة محدودة:
1. قلل BATCH_SIZE
2. استخدم Gradient Accumulation
3. استخدم Mixed Precision Training
"""

import torch
from torch.amp import autocast

def train_with_mixed_precision(model, data, device):
    scaler = torch.cuda.amp.GradScaler()
    
    for batch in data:
        with autocast():  # استخدم float16 تلقائياً
            outputs = model(batch)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# ============================================================================
# 6. مراقبة التدريب مع Tensorboard
# ============================================================================

"""
رؤية الـ Loss والـ Metrics مباشرة
"""

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        # ... التدريب ...
        loss = compute_loss(batch)
        
        # تسجيل الـ Loss
        writer.add_scalar('Training Loss', loss, epoch * len(train_loader) + batch_idx)
        
        # تسجيل Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + batch_idx)

writer.close()

# ثم شغل:
# tensorboard --logdir=runs

# ============================================================================
# 7. استخدام Learning Rate Scheduler
# ============================================================================

"""
تقليل Learning Rate مع الوقت = نموذج أفضل
"""

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# خيار 1: تقليل كل 5 epochs
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# خيار 2: Cosine Annealing (أفضل)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    train()
    scheduler.step()  # تقليل LR

# ============================================================================
# 8. حفظ وتحميل النموذج بشكل آمن
# ============================================================================

"""
الطريقة الصحيحة لحفظ وتحميل
"""

# حفظ
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'tokenizer': tokenizer,
}
torch.save(checkpoint, 'checkpoint.pth')

# تحميل
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# ============================================================================
# 9. تقييم النموذج
# ============================================================================

"""
Metrics للحكم على الجودة:
- Perplexity (أقل = أفضل)
- BLEU Score (أعلى = أفضل)
- ROUGE Score (للملخصات)
"""

import math
from nltk.translate.bleu_score import corpus_bleu

# حساب Perplexity
def calculate_perplexity(loss):
    return math.exp(loss)

val_loss = 2.5
perplexity = calculate_perplexity(val_loss)
print(f"Perplexity: {perplexity:.2f}")

# ============================================================================
# 10. توليد نصوص متنوعة
# ============================================================================

"""
تقنيات مختلفة للتوليد:
1. Greedy: اختر أفضل رمز فقط (سريع، حتمي)
2. Top-K: اختر من أفضل K رمز (متوازن)
3. Top-P: اختر من أفضل P% من الاحتمالات (طبيعي)
4. Sampling: اختر عشوائياً (متنوع)
5. Beam Search: ابحث عن أفضل مسار (جودة عالية)
"""

def generate_variations(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 1. Greedy
    greedy = model.generate(**inputs, max_length=100)
    print("Greedy:", tokenizer.decode(greedy[0]))
    
    # 2. Top-K
    top_k = model.generate(**inputs, max_length=100, top_k=50, do_sample=True)
    print("Top-K:", tokenizer.decode(top_k[0]))
    
    # 3. Top-P
    top_p = model.generate(**inputs, max_length=100, top_p=0.95, do_sample=True)
    print("Top-P:", tokenizer.decode(top_p[0]))
    
    # 4. Beam Search
    beam = model.generate(**inputs, max_length=100, num_beams=5)
    print("Beam Search:", tokenizer.decode(beam[0]))

# ============================================================================
# 11. استخدام API مباشرة (بدون تدريب)
# ============================================================================

"""
إذا ما تبي تدرب نموذج:
استخدم خدمات سحابية أو Hugging Face
"""

# مثال: Hugging Face Inference API
from huggingface_hub import InferenceApi

api = InferenceApi("aubmindlab/aragpt2-base", token="YOUR_TOKEN")
result = api.text_generation("الحمد لله")
print(result)

# ============================================================================
# 12. تحسينات للعربية خاصة
# ============================================================================

"""
نصائح للعربية:
1. استخدم Diacritics (التشكيل) أو بدونها — اختر واحد
2. استخدم Tokenizer عربي متخصص (مش بسيط)
3. الانتباه للتنوين والحروف المتشابهة
4. استخدم نصوص من نفس المصدر (نفس الأسلوب)
"""

import re
from farasa.stemmer import FarasaStemmer

# مثال: Stemming عربي
stemmer = FarasaStemmer()
text = "المؤمنون والمؤمنات"
stemmed = stemmer.stem(text)
print(stemmed)  # سيحول كل شيء للجذر

# ============================================================================
# الخلاصة
# ============================================================================

"""
الخطوات المثالية:
1. جمع بيانات نظيفة ✅
2. استخدام Pre-trained Model ✅
3. Fine-tuning على بياناتك ✅
4. تقييم النموذج ✅
5. تحسين الـ Hyperparameters ✅
6. استخدام Beam Search للتوليد ✅

نصيحة ذهبية:
"Data > Model"
بيانات جيدة مع نموذج بسيط أفضل من
بيانات سيئة مع نموذج معقد
"""
