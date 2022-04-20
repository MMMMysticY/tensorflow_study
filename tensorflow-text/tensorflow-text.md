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
本质还是对vocab文件的映射  
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
