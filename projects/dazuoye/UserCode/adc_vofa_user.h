#ifndef ADC_VOFA_USER_H
#define ADC_VOFA_USER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "adc.h"

/* 用户模块初始化：创建串口互斥锁并配置串口参数 */
void UserAdcVofa_Init(void);
/* ADC1 任务：按 1kS/s 采样 1s，每 5s 启动一次 */
void UserAdcVofa_TaskAdc1(void const *argument);
/* ADC2 任务：按 3kS/s 采样 3s，每 10s 启动一次 */
void UserAdcVofa_TaskAdc2(void const *argument);
/* ADC 转换完成回调：标记窗口采样完成 */
void UserAdcVofa_ConvCpltCallback(ADC_HandleTypeDef *hadc);

#ifdef __cplusplus
}
#endif

#endif /* ADC_VOFA_USER_H */

