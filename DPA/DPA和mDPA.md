## 结论先说

在广告外投语境里，**DPA**通常指 **Dynamic Product Ads，动态商品/动态产品广告**：基于广告主的商品库与用户行为数据，动态生成并投放“用户可能感兴趣的商品”广告，常用于电商商品推荐、召回未转化用户、提升转化。[〔3〕](https://www.52by.com/article/130589)[〔5〕](https://m.toutiao.com/article/7001030301543563807/)[〔11〕](https://zhidao.baidu.com/question/400353081036008085.html)

**mDPA**在腾讯广告等动态商品广告语境中，可理解为 **Multiple Dynamic Product Ads / 多商品动态广告**：与 **SDPA/Single Dynamic Product Ads/单商品动态广告**相对，mDPA 面向“多商品集合/商品库”的动态投放，平台会根据不同用户曾经浏览、点击或表现出兴趣的商品，进行“千人千面”的商品广告展示。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

> 说明：检索结果中也出现了“MDPA = Media Planning and Development Agency”的解释，但它更像“媒体规划推广机构/模式”的说法，与用户问题中的 DPA、SDPA、mDPA 动态商品广告语境不一致；以下按广告外投中的动态商品广告体系解释。[〔1〕](https://zhidao.baidu.com/question/2128012689118813867.html)[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)

---

## 1. DPA 与 mDPA 的含义

| 概念 | 含义 | 核心特征 | 典型用途 |
|---|---|---|---|
| DPA | Dynamic Product Ads，动态商品/动态产品广告 | 基于商品目录和用户历史行为，动态展示当前商品库中的商品 | 商品推荐、再营销、召回未完成转化用户、促进下单 |
| SDPA | Single Dynamic Product Ads，单商品动态广告 | 腾讯广告开发文档中以 `SINGLE` 表示，适用于 SDPA | 单商品型动态投放 |
| mDPA | Multiple Dynamic Product Ads，多商品动态广告 | 腾讯广告开发文档中以 `MULTIPLE` 表示，适用于 MDPA；当动态广告模板为 MDPA 视频模板时，`product_mode` 要传 `MULTIPLE` | 多商品库、千人千面、老客拉活、个性化商品推荐 |

DPA 的本质是“商品库 + 用户行为 + 动态创意/动态推荐”。例如用户在广告主站内或 App 内搜索、点击、收藏、下单但未最终转化，广告主可通过 DPA 在媒体平台上继续向该用户展示其意图商品，从而实现召回和促转化。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)[〔11〕](https://zhidao.baidu.com/question/400353081036008085.html)

mDPA 则更强调“多商品集合”的动态推荐：广告主维护一个可能包含几万到百万商品的商品库，平台根据用户曾经看过或感兴趣的商品，在用户再次出现在媒体环境中时，动态推送其更可能感兴趣的商品广告，实现“千人千面”。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

---

## 2. 产品形式：DPA 与 mDPA 分别长什么样

### 2.1 DPA 的产品形式

DPA 通常是基于商品目录的动态商品广告。广告素材不是完全固定的一张图或一条文案，而是从商品库中动态调用商品信息，并结合用户行为，为不同用户展示不同商品。它适合拥有大量商品、希望针对潜在客户或既有客户做个性化广告的广告主，尤其适合电商类业务。[〔3〕](https://www.52by.com/article/130589)[〔11〕](https://zhidao.baidu.com/question/400353081036008085.html)

从投放逻辑看，DPA 可以展示用户历史浏览、搜索、点击过的商品，也可以展示替代商品或相关商品。例如用户搜索过牛仔外套，系统可以推荐类似款式牛仔外套，也可以推荐牛仔裤等相关商品。[〔11〕](https://zhidao.baidu.com/question/400353081036008085.html)

### 2.2 mDPA 的产品形式

mDPA 是“多商品动态广告”形态，核心是以商品库为基础进行多商品动态匹配。检索结果中腾讯广告开发文档明确将 `product_mode = MULTIPLE` 标明为适用于 MDPA；当 `dynamic_ad_template_id` 传入的模板为 MDPA 视频模板时，`product_mode` 要求强制传 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)

通俗理解，mDPA 会每天更新一个较大的商品集合，并持续监测用户曾经看过哪些商品。当用户再次进入媒体平台环境时，系统向其展示此前感兴趣的商品或相关商品，不同用户看到的商品不同，这就是“千人千面”的多商品动态推荐。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

---

## 3. DPA / mDPA 的通用投放流程

### Step 1：确认准入/开白

检索结果显示，DPA 投放通常需要开白名单；在产研正式对接前，商务侧先进行开白操作。因此，对于广告主来说，第一步通常是确认媒体是否开放 DPA/mDPA 能力、是否需要白名单、由谁发起开白。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

### Step 2：准备商品库/商品目录

DPA/mDPA 的基础是商品目录。腾讯广告开发文档中，动态广告相关接口参数包含 `product_catalog_id`，即商品目录 ID；这说明广告主需要在媒体侧或通过接口维护可被调用的商品目录。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)

### Step 3：同步用户行为数据

广告主需要将用户在广告主端内产生的行为数据同步给投放系统，例如搜索、点击、收藏、下单等。广告主可以基于这些行为数据计算用户的意图商品；如果用户未完成最终转化，就可以通过 DPA 在媒体平台上继续投放其意图商品，进行召回和促转化。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

### Step 4：选择动态商品广告类型

在腾讯广告语境中，需要区分 SDPA 与 MDPA：`product_mode = SINGLE` 表示适用于 SDPA，`product_mode = MULTIPLE` 表示适用于 MDPA；如果使用 MDPA 视频模板，则要求传 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)

### Step 5：媒体侧识别用户并动态出广告

mDPA 的典型逻辑是：广告主有商品库并知道用户对哪些商品感兴趣；当用户再次出现在媒体平台环境中时，媒体平台识别该用户，并向其展示之前感兴趣的商品或相关商品，实现动态推荐和千人千面。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

### Step 6：围绕范围、频次、效果持续优化

广告投放前需要确定广告涉及范围、出现频率和效果，并选择主要媒体形式、具体媒体载体和投放时间安排；同时还需要收集投放地区的媒体信息，如时间、价格、版面、规格等，并分析竞争品牌的投放频率和媒体选择，用于优化投放决策。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)

---

## 4. 广告主需要提供/预估什么

### 4.1 广告主需要提供

1. **商品目录/商品库**：DPA/mDPA 依赖商品目录，腾讯广告动态广告接口中明确包含 `product_catalog_id`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
2. **用户行为数据**：包括搜索、点击、收藏、下单等，用于判断用户意图商品。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)  
3. **广告账户信息**：腾讯广告接口参数中包含 `account_id`，即广告主 ID。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
4. **动态商品广告类型选择**：需要明确使用 SDPA 还是 MDPA；在腾讯广告中对应 `SINGLE` 或 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
5. **模板/创意配置**：如果使用 MDPA 视频模板，腾讯广告文档要求 `product_mode` 强制传 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
6. **投放目标与媒体策略信息**：需要明确投放范围、频次、效果目标，并结合媒体时间、价格、版面、规格等信息进行选择。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)  

### 4.2 广告主需要预估

广告主至少需要预估：广告覆盖范围、投放频次、预期效果、媒体费用、投放时间安排，以及不同媒体形式的优劣。检索结果中的广告投放规范提到，媒体选择要确定广告涉及范围、出现频率和效果，并收集媒体时间、价格、版面、规格等信息。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)

对于 DPA/mDPA，还需要围绕商品库规模、用户行为数据质量、未转化用户规模、可召回用户规模进行评估；其中“用户在端内产生搜索、点击、收藏、下单等行为，并据此计算意图商品、召回未转化用户”是 DPA 流程中的关键基础。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

---

## 5. 媒体侧需要提供/预估什么

### 5.1 媒体侧需要提供

1. **DPA/mDPA 产品能力与准入机制**：检索结果显示 DPA 投放需要开白名单，商务侧先开白再产研对接。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)  
2. **商品目录接入能力**：腾讯广告接口中存在 `product_catalog_id`，说明媒体侧需要支持商品目录接入和调用。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
3. **动态广告类型与模板能力**：腾讯广告支持 `SINGLE`/`MULTIPLE` 两类动态商品广告模式，并存在 MDPA 视频模板场景。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)  
4. **用户识别与动态推荐能力**：mDPA 需要在用户进入媒体环境时识别用户，并根据其历史兴趣商品动态展示广告。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)  
5. **媒体资源与投放信息**：媒体侧通常需要提供或供广告主收集时间、价格、版面、规格等信息，用于广告主进行媒体选择。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)  

### 5.2 媒体侧需要预估

媒体侧需要围绕广告触达范围、出现频率、效果、媒体价格和投放时段等维度提供可评估的信息。检索结果中的媒体选择标准明确提到，投放前要确定广告涉及范围、出现频率和效果，并选择媒体时间安排、媒体形式和具体媒体载体。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)

---

## 6. 中国国内主流媒体的 DPA/mDPA 产品形式与流程

> 受检索结果覆盖范围限制，能够明确确认的平台主要是**腾讯广告**，另有资料提到**今日头条 DPA 投放流程**中的关键步骤；未检索到足够资料来严谨描述其他国内主流媒体的完整 DPA/mDPA 产品形态与流程，因此不做编造。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

### 6.1 腾讯广告：SDPA / mDPA

| 项目 | 内容 |
|---|---|
| 产品形式 | 腾讯广告开发文档中，`product_mode = SINGLE` 表示适用于 SDPA，`product_mode = MULTIPLE` 表示适用于 MDPA。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get) |
| mDPA 形态 | mDPA 可理解为多商品动态广告，适合商品库场景，根据用户历史兴趣进行千人千面推荐。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608) |
| 视频模板 | 当 `dynamic_ad_template_id` 传入的模板为 MDPA 视频模板时，`product_mode` 要求强制传 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get) |
| 核心参数 | 动态广告相关接口参数包含 `account_id`、`product_catalog_id`、`product_mode` 等。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get) |
| 投放逻辑 | 商品库持续更新，平台监测用户曾经看过的商品；当用户再次进入媒体环境时，向其展示感兴趣的商品。[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608) |

腾讯广告 mDPA 的流程可概括为：广告主准备商品目录并在广告账户下接入；选择 `MULTIPLE` 作为多商品动态广告模式；如使用 MDPA 视频模板则强制使用 `MULTIPLE`；媒体侧结合用户历史兴趣和商品库进行动态展示，实现不同用户看到不同商品。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

### 6.2 今日头条：DPA 投放流程

检索结果中提到“今日头条”场景下的 DPA 投放流程，关键步骤包括：先由商务侧进行开白；随后进行产研对接；再同步用户行为数据，如搜索、点击、收藏、下单等；广告主根据这些数据计算用户意图商品；如果用户未完成最终转化，则通过 DPA 在媒体上投放意图商品，召回用户、促进转化。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

该流程体现的是典型 DPA 机制：广告主提供行为数据和意图商品判断，媒体侧承接动态商品广告投放，在媒体流量中触达未转化或潜在转化用户。[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

---

## 7. DPA 与 mDPA 的关键区别

| 维度 | DPA | mDPA |
|---|---|---|
| 定位 | 动态商品广告总称或基础形态 | 多商品动态广告形态 |
| 商品数量 | 可基于商品目录动态展示商品 | 更强调多商品集合/大规模商品库 |
| 用户匹配 | 基于用户历史行为推荐商品 | 针对不同用户动态匹配不同商品，强调千人千面 |
| 平台字段 | 腾讯广告中 SDPA 对应 `SINGLE` | 腾讯广告中 MDPA 对应 `MULTIPLE` |
| 典型场景 | 召回未转化用户、商品推荐、再营销 | 老客拉活、大商品库个性化推荐、多商品动态展示 |

DPA 更像“动态商品广告机制”的统称，而 mDPA 是其中偏多商品集合的一种具体产品模式。在腾讯广告产品语境中，SDPA 与 mDPA 的区分通过 `SINGLE` 和 `MULTIPLE` 表达。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)

---

## 8. 实操建议

如果你是广告主，落地 DPA/mDPA 前建议先确认四件事：第一，媒体是否支持 DPA/mDPA 以及是否需要开白；第二，商品目录是否完整、可更新、可被媒体调用；第三，用户行为数据是否能同步并用于判断意图商品；第四，是选择单商品动态广告还是多商品动态广告，腾讯广告中对应 `SINGLE` 与 `MULTIPLE`。[〔2〕](https://developers.e.qq.com/docs/api/business_assets/dynamic_ad_video/dynamic_ad_video_get)[〔5〕](https://m.toutiao.com/article/7001030301543563807/)

如果你是媒体侧或代理运营方，需要重点核查：是否具备商品目录接入、动态模板、用户识别、行为数据承接、动态推荐和效果评估能力；同时还要提供投放时间、价格、版面/资源规格、覆盖范围、频次和效果等信息，帮助广告主进行媒体选择。[〔4〕](https://wenku.docs.qq.com/detail?docId=ABxd7wfFvv&source=related)[〔8〕](https://www.zhihu.com/question/359253736/answer/2083618608)
