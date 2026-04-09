#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"

#include "adc.h"
#include "tim.h"

#include "stdio.h"

#include "adc_vofa_user.h"


#define ADC_CAPTURE_TIMEOUT_MARGIN_MS  300U     // 300ms 裕量，实际测试中足够覆盖偶发调度延迟
#define VOFA_LINE_BUF_LEN              24U      // 单行 FireWater 文本缓存长度
#define USB_CDC_TX_TIMEOUT_MS          20U      // 单帧发送超时，避免BUSY时长时间阻塞

#define USBD_OK_VALUE                  0U
#define USBD_BUSY_VALUE                1U

// ADC1 采样参数
#define ADC1_SAMPLE_RATE_HZ            1000U    // 采样率 1kS/s
#define ADC1_CAPTURE_WINDOW_MS         1000U    // 采样窗口 1s
#define ADC1_CAPTURE_INTERVAL_MS       5000U    // 窗口周期 5s
// 采样点数 1000 * 1s = 1000 点
#define ADC1_SAMPLE_COUNT              ((ADC1_SAMPLE_RATE_HZ * ADC1_CAPTURE_WINDOW_MS) / 1000U) 

// ADC2 采样参数
#define ADC2_SAMPLE_RATE_HZ            3000U    // 采样率 3kS/s
#define ADC2_CAPTURE_WINDOW_MS         3000U    // 采样窗口 3s
#define ADC2_CAPTURE_INTERVAL_MS       10000U   // 窗口周期 10s
// 采样点数 3000 * 3s = 9000 点
#define ADC2_SAMPLE_COUNT              ((ADC2_SAMPLE_RATE_HZ * ADC2_CAPTURE_WINDOW_MS) / 1000U)

// ADC 原始采样缓冲区
static uint16_t g_adc1_buffer[ADC1_SAMPLE_COUNT];
static uint16_t g_adc2_buffer[ADC2_SAMPLE_COUNT];

// DMA 窗口采样完成标志,中断里置位，任务里轮询等待
static volatile uint8_t g_adc1_capture_done = 0U;
static volatile uint8_t g_adc2_capture_done = 0U;

// 最近一次发送值：当某一通道正在发送窗口数据时，另一通道用最近值补齐
static uint16_t g_adc1_latest = 0U;
static uint16_t g_adc2_latest = 0U;

// 发送互斥锁：保护 CDC_Transmit_FS，防止两任务并发输出冲突
static osMutexId g_uart_tx_mutex = NULL;

// 定义名为UserVofaUartMutex的互斥锁对象
osMutexDef(UserVofaUartMutex);

/*
 * 弱定义兜底：
 * 在 CubeMX 未启用 USB_DEVICE 生成 usbd_cdc_if.c 前，工程也可链接通过。
 * 启用 USB CDC 后，强定义的 CDC_Transmit_FS 会覆盖本函数。
 */
__weak uint8_t CDC_Transmit_FS(uint8_t *Buf, uint16_t Len)
{
  (void)Buf;
  (void)Len;
  return 2U; // USBD_FAIL
}

// 阻塞式 CDC 发送，内部处理 USBD_BUSY 重试
static HAL_StatusTypeDef UserAdcVofa_SendBytes(uint8_t *tx_buf, uint16_t tx_len)
{
  uint32_t start_tick;
  uint8_t tx_state;

  start_tick = HAL_GetTick();

  for (;;)
  {
    tx_state = CDC_Transmit_FS(tx_buf, tx_len);
    if (tx_state == USBD_OK_VALUE)
    {
      return HAL_OK;
    }
    if (tx_state != USBD_BUSY_VALUE)
    {
      return HAL_ERROR;
    }

    if ((HAL_GetTick() - start_tick) > USB_CDC_TX_TIMEOUT_MS)
    {
      return HAL_TIMEOUT;
    }
    osDelay(1);
  }
}


// 使用阻塞发送一帧 FireWater 文本：adc1,adc2\r\n
// adc1_raw/adc2_raw: 当前采样点的原始 ADC 值
static void UserAdcVofa_SendFrame(uint16_t adc1_raw, uint16_t adc2_raw)
{
  char tx_line[VOFA_LINE_BUF_LEN];  // 发送缓冲数组
  int line_len;                     // snprintf实际生成的字符串长度

  // 两个 ADC 数值按格式拼成字符串写入 tx_line
  line_len = snprintf(tx_line, sizeof(tx_line), "%u,%u\r\n", adc1_raw, adc2_raw);

  if (line_len > 0)
  {
    // USB CDC发送一帧文本
    (void)UserAdcVofa_SendBytes((uint8_t *)tx_line, (uint16_t)line_len);
  }
}

/*
 * 启动一次窗口采样（DMA + 定时器触发）并等待结束。
 * 参数：
 *   hadc         目标 ADC 句柄
 *   htim         对应触发定时器句柄（TIM2/TIM3）
 *   sample_buffer DMA 目标缓存
 *   sample_count  本次窗口采样点数
 *   done_flag     完成标志地址（由中断回调置位）
 *   timeout_ms    最大等待时长
 * 返回：
 *   HAL_OK / HAL_ERROR / HAL_TIMEOUT
 */
static HAL_StatusTypeDef UserAdcVofa_CaptureWindow(ADC_HandleTypeDef *hadc,
                                                    TIM_HandleTypeDef *htim,
                                                    uint16_t *sample_buffer,
                                                    uint32_t sample_count,
                                                    volatile uint8_t *done_flag,
                                                    uint32_t timeout_ms)
{
  uint32_t start_tick;

  /* 采样前清零完成标志，并复位定时器计数器 */
  *done_flag = 0U;
  __HAL_TIM_SET_COUNTER(htim, 0U);

  /* 先启动 ADC DMA，再开启定时器触发，保证触发到来时 DMA 已就绪 */
  if (HAL_ADC_Start_DMA(hadc, (uint32_t *)sample_buffer, sample_count) != HAL_OK)
  {
    return HAL_ERROR;
  }

  if (HAL_TIM_Base_Start(htim) != HAL_OK)
  {
    (void)HAL_ADC_Stop_DMA(hadc);
    return HAL_ERROR;
  }

  /* 轮询等待窗口采样完成（完成标志由中断回调置位） */
  start_tick = HAL_GetTick();
  while (*done_flag == 0U)
  {
    if ((HAL_GetTick() - start_tick) > timeout_ms)
    {
      /* 超时保护：停止触发与 DMA，防止资源被占用 */
      (void)HAL_TIM_Base_Stop(htim);
      (void)HAL_ADC_Stop_DMA(hadc);
      return HAL_TIMEOUT;
    }
    /* 让出 CPU，避免忙等 */
    osDelay(1);
  }

  /* 正常完成后，确保定时器和 DMA 处于停止状态 */
  (void)HAL_TIM_Base_Stop(htim);
  (void)HAL_ADC_Stop_DMA(hadc);
  return HAL_OK;
}

/* 发送 ADC1 一个窗口的数据，ADC2 列使用最近值补齐 */
static void UserAdcVofa_SendAdc1Window(void)
{
  uint32_t i;

  /* 整个窗口发送期间持有互斥锁，保证串口输出连续且不被另一任务打断 */
  if (g_uart_tx_mutex != NULL)
  {
    (void)osMutexWait(g_uart_tx_mutex, osWaitForever);
  }

  for (i = 0U; i < ADC1_SAMPLE_COUNT; ++i)
  {
    /* ADC1 更新为当前采样点，ADC2 使用其最近值 */
    g_adc1_latest = g_adc1_buffer[i];
    UserAdcVofa_SendFrame(g_adc1_latest, g_adc2_latest);
  }

  if (g_uart_tx_mutex != NULL)
  {
    (void)osMutexRelease(g_uart_tx_mutex);
  }
}

/* 发送 ADC2 一个窗口的数据，ADC1 列使用最近值补齐 */
static void UserAdcVofa_SendAdc2Window(void)
{
  uint32_t i;

  /* 整个窗口发送期间持有互斥锁，保证串口输出连续且不被另一任务打断 */
  if (g_uart_tx_mutex != NULL)
  {
    (void)osMutexWait(g_uart_tx_mutex, osWaitForever);
  }

  for (i = 0U; i < ADC2_SAMPLE_COUNT; ++i)
  {
    /* ADC2 更新为当前采样点，ADC1 使用其最近值 */
    g_adc2_latest = g_adc2_buffer[i];
    UserAdcVofa_SendFrame(g_adc1_latest, g_adc2_latest);
  }

  if (g_uart_tx_mutex != NULL)
  {
    (void)osMutexRelease(g_uart_tx_mutex);
  }
}

/*
 * 用户模块初始化：
 * 1) 创建发送互斥锁
 */
void UserAdcVofa_Init(void)
{
  g_uart_tx_mutex = osMutexCreate(osMutex(UserVofaUartMutex));
}

/*
 * 任务1：ADC1 周期采样与发送
 * 采样率 1kS/s，窗口 1s，周期 5s。
 */
void UserAdcVofa_TaskAdc1(void const *argument)
{
  TickType_t last_wake_tick;

  (void)argument;
  /* 记录当前节拍，配合 vTaskDelayUntil 实现稳定周期调度 */
  last_wake_tick = xTaskGetTickCount();

  for (;;)
  {
    /* 采样成功后，发送 ADC1 窗口数据 */
    if (UserAdcVofa_CaptureWindow(&hadc1,
                                  &htim2,
                                  g_adc1_buffer,
                                  ADC1_SAMPLE_COUNT,
                                  &g_adc1_capture_done,
                                  ADC1_CAPTURE_WINDOW_MS + ADC_CAPTURE_TIMEOUT_MARGIN_MS) == HAL_OK)
    {
      UserAdcVofa_SendAdc1Window();
    }

    /* 固定周期调度：从上次唤醒点起每 5s 运行一次 */
    vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(ADC1_CAPTURE_INTERVAL_MS));
  }
}

/*
 * 任务2：ADC2 周期采样与发送
 * 采样率 3kS/s，窗口 3s，周期 10s。
 */
void UserAdcVofa_TaskAdc2(void const *argument)
{
  TickType_t last_wake_tick;

  (void)argument;
  /* 记录当前节拍，配合 vTaskDelayUntil 实现稳定周期调度 */
  last_wake_tick = xTaskGetTickCount();

  for (;;)
  {
    /* 采样成功后，发送 ADC2 窗口数据 */
    if (UserAdcVofa_CaptureWindow(&hadc2,
                                  &htim3,
                                  g_adc2_buffer,
                                  ADC2_SAMPLE_COUNT,
                                  &g_adc2_capture_done,
                                  ADC2_CAPTURE_WINDOW_MS + ADC_CAPTURE_TIMEOUT_MARGIN_MS) == HAL_OK)
    {
      UserAdcVofa_SendAdc2Window();
    }

    /* 固定周期调度：从上次唤醒点起每 10s 运行一次 */
    vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(ADC2_CAPTURE_INTERVAL_MS));
  }
}

/*
 * ADC 转换完成回调（由 HAL 在中断上下文调用）。
 * 注意：中断回调内只做短操作（置位标志、关定时器），不做阻塞调用。
 */
void UserAdcVofa_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
  if (hadc->Instance == ADC1)
  {
    /* ADC1 窗口采满：停止触发定时器并置完成标志 */
    __HAL_TIM_DISABLE(&htim2);
    g_adc1_capture_done = 1U;
  }
  else if (hadc->Instance == ADC2)
  {
    /* ADC2 窗口采满：停止触发定时器并置完成标志 */
    __HAL_TIM_DISABLE(&htim3);
    g_adc2_capture_done = 1U;
  }
  else
  {
    /* 非目标 ADC，忽略 */
  }
}
