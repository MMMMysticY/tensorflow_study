- [tensorflow-text部分目录](#tensorflow-text部分目录)
  - [unicode](#unicode)
    - [unicode的tf表示](#unicode的tf表示)
    - [tf.strings处理unicode](#tfstrings处理unicode)
    - [对带有batch的内容处理 RaggedTensor](#对带有batch的内容处理-raggedtensor)
    - [对unicode对象的操作](#对unicode对象的操作)
    - [获得unicode表示的character所在的语言](#获得unicode表示的character所在的语言)
  - [word embedding](#word-embedding)
  - [decoding解码方式](#decoding解码方式)
  - [PRE-PROCESSING 文本预处理](#pre-processing-文本预处理)
    - [tokenize string对string进行token分隔](#tokenize-string对string进行token分隔)
      - [tokenize的API](#tokenize的api)
      - [Whole word tokenizers 整个单词进行tokenize](#whole-word-tokenizers-整个单词进行tokenize)
      - [Subword tokenizers](#subword-tokenizers)
      - [其他Tokenizer](#其他tokenizer)
      - [offset](#offset)
      - [detokenization](#detokenization)
      - [对tf.data.Dataset对象使用tokenizer](#对tfdatadataset对象使用tokenizer)
    - [subword tokenizer](#subword-tokenizer)
      - [通过tf.data.Dataset生成bert_vocab](#通过tfdatadataset生成bert_vocab)
      - [基于vocab生成BertTokenizer](#基于vocab生成berttokenizer)
    - [BERT preprocessing](#bert-preprocessing)
  - [BERT Experiments](#bert-experiments)
# tensorflow-text部分目录

## unicode
见[unicode part](basic_concepts/work_with_unicode.ipynb)  
NLP模型经常处理不同的语言，不同的语言又有不同的词典  Unicode是针对于几乎所有语言都可以用其进行表示文字的方法  
unicode character是0-0x0FFFF的int值 unicode string是一串0或者unicode character  

### unicode的tf表示
1. 直接通过constant表示 这样的表示的结果是一个byte形式 b'\ \ \'的形式
```python
tf.constant(u"abc")
tf.constant([u"abc", u"def"])
```
2. 通过unicode codepoint表示
```python
tf.constant([ord(s) for s in u"自然语言处理"])
```
### tf.strings处理unicode
1. tf.strings.unicode_decode:将string scalar转为code points的向量
```python
text_utf8 = tf.constant("自然语言处理") # <tf.Tensor: shape=(), dtype=string, numpy=b'\xe8\x87\xaa\xe7\x84\xb6\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'>
tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8') # <tf.Tensor: shape=(6,), dtype=int32, numpy=array([33258, 28982, 35821, 35328, 22788, 29702], dtype=int32)>
```
2. tf.strings.unicode_encode:将code points向量转为 string scalar
```python
text_chars = tf.constant([ord(s) for s in u"自然语言处理"]) # <tf.Tensor: shape=(6,), dtype=int32, numpy=array([33258, 28982, 35821, 35328, 22788, 29702], dtype=int32)>
tf.strings.unicode_encode(text_chars, output_encoding='UTF-8') # <tf.Tensor: shape=(), dtype=string, numpy=b'\xe8\x87\xaa\xe7\x84\xb6\xe8\xaf\xad\xe8\xa8\x80\xe5\xa4\x84\xe7\x90\x86'>
```
3. tf.strings.unicode_transcode:将string scalar转为其他形式的编码 如utf-8 -> utf-16-be
```python
text_utf16be = tf.constant(u"自然语言处理".encode("UTF-16-BE")) # <tf.Tensor: shape=(), dtype=string, numpy=b'\x81\xeaq6\x8b\xed\x8a\x00Y\x04t\x06'>
tf.strings.unicode_transcode(text_utf16be, input_encoding='UTF-16-BE', output_encoding='UTF-8') # tf.strings.unicode_transcode(text_utf16be, input_encoding='UTF-16-BE', output_encoding='UTF-8')
```

### 对带有batch的内容处理 RaggedTensor
```python
batch_utf8 = [s.encode('UTF-8') for s in [u'hÃllo', u'What is the weather tomorrow', u'Göödnight', u'😊']]
'''
[b'h\xc3\x83llo',
 b'What is the weather tomorrow',
 b'G\xc3\xb6\xc3\xb6dnight',
 b'\xf0\x9f\x98\x8a']
'''
batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding='UTF-8')
'''
<tf.RaggedTensor [[104, 195, 108, 108, 111],
 [87, 104, 97, 116, 32, 105, 115, 32, 116, 104, 101, 32, 119, 101, 97, 116,
  104, 101, 114, 32, 116, 111, 109, 111, 114, 114, 111, 119]               ,
 [71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]]>
'''
```

### 对unicode对象的操作
1. character length 通过不同的unit统计unicode的长度有多少units tf.strings.length
2. character substrings 取unicode的substring tf.strings.substr
3. split unicode strings 对unicode string进行切断 tf.strings.unicode_split
4. byte offset for characters  获得byte offset tf.strings.unicode_decode_with_offsets

### 获得unicode表示的character所在的语言
```python
# unicode 33464代表汉字芸 1041代表西里尔语Б
# 可以直接处理list对象
uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']
print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]
# unicode_script之后得到了17->汉语 8->西里尔语
```

## word embedding
见[word embedding](basic_concepts/word_embedding.ipynb)  
word embedding的核心思想是将token转化为一个稠密的向量表示，进而就能进行网络训练 还可以查看token的相关性 相似度等  
注意两个API:
1. TextVectorization keras.layers.TextVectorization 通过文本的训练得到一个从token映射到int值的表 之后通过其计算，可以将文本token组成的句子变成int向量
2. Embedding keras.layers.Embedding 将int向量表示的句子，转化为稠密向量表示 进行后续网络的作用

## decoding解码方式
见[Decoding_methods](basic_concepts/decoding.ipynb)  
解码方式有：
1. Greedy
2. Beam Search Beam search通过搜索每个时间步概率最高的num_beams进行后续搜索，这样防止在过程中丢弃总体概率最高的结果
3. Top-k top-k采样仅采用概率最高的K个内部进行概率重新分配 进行生成
4. Top-p top-p采样使用累计概率和超过p的最小的概率作为结果

在from official.nlp.modeling.ops import sampling_module 和 from official.nlp.modeling.ops import beam_search 中有高度集成的API  


## PRE-PROCESSING 文本预处理

text preprocessing是一种将原始文本转化为可以输入模型的int形式的向量  
文本预处理不佳的危害有三点：
1. **Training-serving skew 训练-服务的差异**。 如果在训练和使用的各个不同阶段使用不同的超参数 tokenization token的方法和预处理的方法，会给模型带来毁灭性的效果。
2. **Efficiency and flexibility 高效可扩展**。当preprocessing离线进行时(将处理后的输出写到磁盘中，之后重新读入)这样会产生额外的成本；如果需要动态进行预处理决策 离线preprocessing也不合适。
3. **Complex model interface 复杂模型推理**。当文本模型的输入是纯文本时，模型更易于处理。当模型输入需要额外的编码步骤时，模型就不易理解。降低预处理复杂性对调试、服务和评估很重要。

### tokenize string对string进行token分隔
见[tokenize strings](pre-processing/tokenize_strings.ipynb)  
#### tokenize的API
主要的接口是Splitter和SplitterWithOffset API 他们分别只有一个方法split 和 split_with_offset  
**Tokenizer和TokenizerWithOffset是Splitter的实例化 提供了便捷的tokenize和tokenize_with_offsets方法**  
输入是N维矩阵的话输出是N+1维的RaggedTensor 最内层是tokenize的结果  
同时还有Detokenizer接口 通过tokenizer接口tokenize的N维RaggedTensor会变为N-1维的tensor或者RaggedTensor  
#### Whole word tokenizers 整个单词进行tokenize
1. WhitespaceTokenizer  
空白符号tokenizer使用ICU定义的空白字符(空格 制表符 换行符)进行分割   
**空白符号tokenizer将标点符号和字母连接在了一起 同时不能处理类似于汉字没有空格分隔的句子**   
2. UnicodeScriptTokenizer   
UnicodeScriptTokenizer用unicode进行分割 同时将标点符号单独处理   
**UnicodeScriptTokenizer还是使用空白符进行分割 不能分割没有空白符的汉字** 但是能将标点符号处理   
#### Subword tokenizers
subword tokenizers可以和较小的词汇表一起使用，允许模型从创建词汇的子词中获得一些信息  
1. WordpieceTokenizer  
WordpieceTokenizer是一种数据驱动的tokenization方法 其生成了一组sub-tokens 和语言本身的语素有关  
WordpieceTokenizer期望接受一个分割后的tokens作为输入进行数据驱动 所以一般先用WhiteSpaceTokenizer或者UnicodeScriptTokenizer  
本质就是对vocab文件的映射  
2. BertTokenizer
BertTokenizer实现了BERT论文中的实现方法，本质是由WordPieceTokenizer支持的 但是还执行其他任务 如单词规范化和标记化  
值得注意的是BertTokenizer中有可能对一个word进行分割到sub-word，例如"Average" -> "A" "##ven" "##ger"  
这是因为vocab表中有词根的概念 ##+characters 表示了一个词根 那么就可以将一个词进行分割  
输入是[batch, num_tokens(一个句子中词的个数)]  
输出是[batch, num_tokens(一个句子中词的个数 这个和输入完全相同), num_wordpieces(**这个维度是每个word用分成了几维 有可能是1 即没有分割为sub-word 但是也会被分成词根 那么就是多维**)]  
部分任务需要保留num_wordpieces维，大部分任务不需要，就可以直接将其合并为[batch, new_num_tokens] 方法为merge_dims(-2,-1)  
本质还是对vocab文件的映射(注意词根 ##+字母的模式)  
3. SentencepieceTokenizer
SentencepieceTokenizer是基于sentencepiece库的 这个方法是根据输入数据快速迭代的 有很明显的sub-word的效果 见ML_tools仓库[ML_tools](https://github.com/MMMMysticY/ML_tools/tree/master/NLP/sentencepiece)
#### 其他Tokenizer
1. UnicodeCharTokenizer  
按照UTF-8进行分隔 对没有空格的语言很有用  
本质和[word_with_unicode](basic_concepts/work_with_unicode.ipynb)中一样 用unicode编号进行tokenize  
2. HubModuleTokenizer  
这是一个部署在TF Hub上的模型 并不支持RaggedTensor 这个对没有空格的启发式语言很有效果  
**这个很棒！https://hub.tensorflow.google.cn/google/zh_segmentation/1 提供了一个应用于中文的按照语义进行分割的tokenizer**  
3. SplitMergeTokenizer  
SplitMergeTokenizer 和 SplitMergeFromLogitsTokenizer 通过对分割点的显式提供进行分割  
SplitMergeTokenizer 提供0,1的向量0代表分割点即token的开头1代表不分隔  
SplitMergeFromLogitsTokenizer 是通过得分进行分割 第一维大于第二维则代表0 否则代表1  
4. RegexSplitter  
通过正则表达式进行确定分割点  

#### offset
如果要知道每个被分隔的token在原始句子的位置 就可以使用几乎所有tokenizer的tokenize_with_offsets方法  
左闭右开的  

#### detokenization
tokenize的逆操作 但是并不是所有的tokenizer都有这个方法  
同时**tokenize和detokenization可能是有损的 并不一定完全还原**  

#### 对tf.data.Dataset对象使用tokenizer
使用map(lambda x: tokenizer.tokenize(x))方法  

### subword tokenizer
有几个实用的方法：
#### 通过tf.data.Dataset生成bert_vocab
```python
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
pt_vocab = bert_vocab.bert_vocab_from_dataset()
```
见[bert_vocab](pre-processing/subword_tokenizers.ipynb)  
#### 基于vocab生成BertTokenizer
这个和上一部分subword小节中BertTokenizer一样  
但是值得注意几个方法  
```python
tf.gather(vocab_txt, token_int) # 这个方法可以简单地将int值映射到vocab_txt文件中的字符上
tf.strings.reduce_join(text_tokens, separator=' ', axis=-1) # 这个方法可以将维度内的各个值合并 以空格分隔
(RaggedTensor).merge_dims(-2,-1) # 这个方法很有用 因为BERT tokenize之后的结果是一个RaggedTensor是 [batch, seq_len, N] 最后这个N很多情况下等于1在有些情况没有意义 可以直接合并

bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
# 这个方法可以进行正则匹配删去特定tokens
```
还有CustomTokenizer方法  
见[BertTokenizer后续处理](pre-processing/subword_tokenizers.ipynb)  

### BERT preprocessing
使用丰富的tensorflow_text的api 完成BERT任务 即Masked language model + next sentence prediction  
基本方法是：
1. Input是tf.string类型的**Tensor** 其中有text_a text_b  
   维度：每个text_a和text_b的维度是[batch, 1]  
   方法：tf.data.Dataset.from_tensors   
2. 将Input进行Tokenizer **按照vocab初始化BertTokenizer**  
   维度：每个text_a text_b 维度变成[batch, num_words, wordpieces] 即将一个句子分成word 再把word分成wordpieces  
   方法：tf.lookup.StaticVocabularyTable初始化lookup对象， 作为参数初始化text.BertTokenizer 之后调用tokenize方法  
3. 本任务无需wordpieces单独处理 所以将每个text_a text_b最后两维合并  
   维度：text_a text_b维度变成[batch, num_wordpieces] 
   方法：merge_dims(-2,-1)  
4. 将两个句子裁剪为MAX_SEQ_LEN以内  
   维度：text_a变成[batch, num_wordpieces_a] text_b变成[batch, num_wordpiece_b] 其中num_wordpieces_a + num_wordpiece_b <= MAX_SEQ_LEN  
   方法：text.RoundRobinTrimmer trim方法  
5. 将text_a text_b 进行拼接 并加上SOS和EOS  
   维度：输出的combined_segments是text_a和text_b合并的整体结果 [batch, seq_len] seq_len = 3 + num_wordpieces_a + num_wordpiece_b  同时还有一个输出segment_ids进行text_a和text_b的区分，0代表text_a 1代表text_b 维度也是[batch, seq_len]  
   方法：text.combine_segments  
6. 进行mask 随机选择mask的位置和mask的value 进行mask  
   维度：masked_input_ids是combined_segments进行mask的结果 维度是[batch, seq_len] 不变 因为只有mask行为 masked_positions和masked_ids是被mask的位置和原始真实的ids，维度是[batch, masked_len]  
   方法：mask位置的随机选择器text.RandomItemSelector mask位置的处理方法值的选取text.MaskValuesChooser 进行mask的方法text.mask_language_model  
7. 进行pad到MAX_SEQ_LEN 和 MAX_PREDICTION_LEN  
   维度：masked_input_ids 和 segment_ids pad到 MAX_SEQ_LEN masked_positions和masked_ids pad到MAX_PREDICTION_LEN  
   方法：text.pad_model_inputs  
8. 用上面所有东西作为inputs

详细见[BERT-preprocessing](pre-processing/BERT-preprocessing.ipynb)  

## BERT Experiments
使用BERT进行基本的fine-tune任务的步骤是：
1. **获取处理数据集** 下载或者加载 这个过程有很多技巧 例如
   - keras.utils.text_dataset_from_directory可以直接加载txt文本(见[text_classification_with_BERT](tensorflow-text/bert_exp/text_classification_with_BERT.ipynb))
   - tensorflow_datasets库tfds的load方法 可以下载也可以从本地加载(**下载基本在国内不行 所以下载了之后加载比较合适**)(见[Fine tune BERT](tensorflow-text/bert_exp/Fine%20tune%20BERT.ipynb))
   - 下载了TFRecordDataset对象之后 使用tf.io.FixedLenFeature+tf.io.parse_single_example进行解析(**这个似乎在大规模数据集比较合适**)(见[Fine tune BERT](tensorflow-text/bert_exp/Fine%20tune%20BERT.ipynb))
2. **处理数据集** 处理数据集有很多方法
   - 下载tf hub的BERT对应的preprocess直接进行处理bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)(见[text_classification_with_BERT](tensorflow-text/bert_exp/text_classification_with_BERT.ipynb))
   - 使用official.nlp库的方法bert.tokenization.FullTokenizer再加上一些原生方法如RaggedTensor.to_tensor()进行简单的pad或者keras.preprocessing.sequence.pad_sequences进行pad (见[Fine tune BERT](tensorflow-text/bert_exp/Fine%20tune%20BERT.ipynb))
3. **构建模型** 使用BERT预训练的模型作为pre-train部分 当做一个encoder 其会输出四个结果 (见[text_classification_with_BERT](tensorflow-text/bert_exp/text_classification_with_BERT.ipynb))
   - default default是默认的也就是pooled_output
   - pooled_output 将input sequence表示为一个整体 维度是[batch_size, hidden] 这是整个句子的一个embedding
   - sequence output 表示了每个token的embedding 维度是[batch_size, seq_len, hidden] 是每个token的embedding
   - encoder_output 表示encoder中每个Transformer blocks的中间隐层状态 [block_num, batch_size, seq_len, hidden] block_num代表了transformer encoder的个数 encoder_outputs[-1] == sequence_output
  
    选择合适的embedding进行后续任务 后续任务就是Model的class compile fit等
4. 保存模型 tf.saved_model.save或者model.save 
5. 加载模型 tf.saved_model.load 这里有个问题(如何定义或者修改模型的Input，在[Fine tune BERT](tensorflow-text/bert_exp/Fine%20tune%20BERT.ipynb)遇到过一些问题 save之后再load维度不对)
6. 使用nlp.data.classifier_data_lib.TfdsProcessor和TFRecord进行大型数据集的encoding(由于网络问题没做)(见[Fine tune BERT](tensorflow-text/bert_exp/Fine%20tune%20BERT.ipynb))
