"""
Text Generation Model - Complete Pipeline
Islamic Arabic Text Generation with PyTorch
Author: Claude (for Ismail Abu Muawiyah)
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
from pathlib import Path
import json

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

class TextPreprocessor:
    """تنظيف ومعالجة النصوص العربية"""
    
    @staticmethod
    def clean_text(text):
        """إزالة التشكيل والرموز غير المفيدة"""
        # إزالة التشكيل (diacritics)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        # إزالة الأرقام الإنجليزية
        text = re.sub(r'[0-9]', '', text)
        # الاحتفاظ بالعربية والمسافات فقط
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def load_texts(file_path):
        """تحميل النصوص من ملف"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = TextPreprocessor.clean_text(line)
                if len(cleaned) > 10:  # تجاهل النصوص القصيرة جداً
                    texts.append(cleaned)
        return texts
    
    @staticmethod
    def save_cleaned_texts(texts, output_path):
        """حفظ النصوص المنظفة"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        print(f"✅ تم حفظ {len(texts)} نص في {output_path}")

# ============================================================================
# 2. TOKENIZER
# ============================================================================

class SimpleArabicTokenizer:
    """Tokenizer بسيط للعربية"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<eos>'}
    
    def build_vocab(self, texts):
        """بناء قاموس من النصوص"""
        word_freq = {}
        
        # حساب تكرار الكلمات
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # ترتيب أكثر الكلمات تكراراً
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # بناء القاموس
        idx = len(self.word2idx)
        for word, freq in sorted_words:
            if idx < self.vocab_size:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
            else:
                break
        
        print(f"✅ تم بناء قاموس بـ {len(self.word2idx)} كلمة")
    
    def encode(self, text):
        """تحويل نص لأرقام"""
        words = text.split()
        return [self.word2idx.get(w, 1) for w in words]  # 1 = <unk>
    
    def decode(self, indices):
        """تحويل أرقام لنص"""
        return ' '.join([self.idx2word.get(i, '<unk>') for i in indices])
    
    def save(self, path):
        """حفظ الـ Tokenizer"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False)
    
    def load(self, path):
        """تحميل الـ Tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}

# ============================================================================
# 3. DATASET
# ============================================================================

class TextDataset(Dataset):
    """Dataset لنصوص التدريب"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        # تحويل النصوص لـ sequences
        for text in texts:
            tokens = self.tokenizer.encode(text)
            # تقسيم لـ overlapping sequences
            for i in range(len(tokens) - 1):
                self.sequences.append(tokens[i:i+self.max_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Padding
        if len(seq) < self.max_length:
            seq = seq + [0] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        
        return torch.LongTensor(seq)

# ============================================================================
# 4. MODELS
# ============================================================================

class LSTMTextGenerator(nn.Module):
    """نموذج LSTM بسيط"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

class TransformerTextGenerator(nn.Module):
    """نموذج Transformer (أفضل)"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_positional_encoding(max_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def _create_positional_encoding(self, max_length, d_model):
        """إنشاء positional encoding"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.transformer(x)
        logits = self.fc(x)
        return logits

# ============================================================================
# 5. TRAINING
# ============================================================================

class TextGenerationTrainer:
    """مدرب النموذج"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = CrossEntropyLoss(ignore_index=0)
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """حقبة واحدة من التدريب"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            
            # Loss (التنبؤ بالكلمة التالية)
            loss = self.loss_fn(
                logits[:, :-1].reshape(-1, self.model.fc.out_features),
                batch[:, 1:].reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """التقييم على مجموعة التحقق"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.loss_fn(
                    logits[:, :-1].reshape(-1, self.model.fc.out_features),
                    batch[:, 1:].reshape(-1)
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """تدريب النموذج"""
        print(f"\n{'='*60}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            
            # حفظ أفضل نموذج
            # if epoch == 0 or val_loss < min(self.val_losses[:-1]):
            #     self.save_checkpoint("best_model.pth")
            #     print("  💾 نموذج جديد محفوظ!")

# ============================================================================
# 6. GENERATION
# ============================================================================

class TextGenerator:
    """توليد النصوص"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt, max_length=50, temperature=0.7, top_k=5):
        """توليد نص من prompt"""
        self.model.eval()
        
        # تشفير الـ prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor([tokens]).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # التنبؤ بالكلمة التالية
                logits = self.model(input_ids)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(top_k_probs, 1)
                next_token = top_k_indices[0, next_token_idx]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # فك التشفير
        generated_tokens = input_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text

# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    """Pipeline كامل"""
    
    # إعدادات
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    VOCAB_SIZE = 10000
    MAX_LENGTH = 128
    
    print(f"🖥️  Device: {DEVICE}")
    print(f"📊 Batch Size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    
    # ============================================================================
    # خطوة 1: تحضير البيانات
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 1: Data Preparation")
    print("="*60)
    
    # اختر واحد من الخيارات:
    # 1. استخدم ملف موجود
    DATA_FILE = "NLP/islamic_texts.txt"  # ضع ملف البيانات هنا
    
    if not Path(DATA_FILE).exists():
        print(f"⚠️  الملف {DATA_FILE} غير موجود!")
        print("   أنشئ ملف بهذا الاسم يحتوي على نصوص عربية (سطر واحد لكل نص)")
        return
    
    print(f"📖 تحميل النصوص من {DATA_FILE}...")
    texts = TextPreprocessor.load_texts(DATA_FILE)
    print(f"✅ تم تحميل {len(texts)} نصوص")
    print(f"   مثال: {texts[0][:100]}...")
    
    # تقسيم البيانات
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"📊 التدريب: {len(train_texts)}, التحقق: {len(val_texts)}")
    
    # ============================================================================
    # خطوة 2: بناء Tokenizer
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 2: Building Tokenizer")
    print("="*60)
    
    tokenizer = SimpleArabicTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(train_texts)
    tokenizer.save("tokenizer.json")
    
    # ============================================================================
    # خطوة 3: إنشاء Datasets
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 3: Creating Datasets")
    print("="*60)
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=MAX_LENGTH)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"✅ Train Dataset: {len(train_dataset)} sequences")
    print(f"✅ Val Dataset: {len(val_dataset)} sequences")
    
    # ============================================================================
    # خطوة 4: بناء النموذج
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 4: Building Model")
    print("="*60)
    
    # اختر بين LSTM و Transformer
    MODEL_TYPE = "lstm"  # أو "lstm"
    
    if MODEL_TYPE == "lstm":
        model = LSTMTextGenerator(vocab_size=VOCAB_SIZE).to(DEVICE)
    else:
        model = TransformerTextGenerator(vocab_size=VOCAB_SIZE).to(DEVICE)
    
    print(f"✅ Model: {MODEL_TYPE.upper()}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============================================================================
    # خطوة 5: التدريب
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 5: Training")
    print("="*60)
    
    trainer = TextGenerationTrainer(model, DEVICE, learning_rate=0.001)
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # ============================================================================
    # خطوة 6: التوليد
    # ============================================================================
    print("\n" + "="*60)
    print("STEP 6: Text Generation")
    print("="*60)
    
    generator = TextGenerator(model, tokenizer, DEVICE)
    
    # أمثلة للتوليد
    prompts = [
        "الحمد لله",
        "يا ايها الناس",
        "قال الله تعالى"
    ]
    
    for prompt in prompts:
        generated = generator.generate(prompt, max_length=50, temperature=0.7)
        print(f"\n📝 Prompt: {prompt}")
        print(f"   Generated: {generated}")
    
    # حفظ النموذج النهائي
    torch.save(model.state_dict(), f"text_generator_{MODEL_TYPE}.pth")
    print(f"\n✅ النموذج محفوظ في text_generator_{MODEL_TYPE}.pth")

if __name__ == "__main__":
    main()
