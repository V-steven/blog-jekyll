---
layout: post
title: keyCallback model
tags:  [嵌入式]
excerpt: "按键与执行函数绑定，同时使用执行周期进行按键延时                                          "
---
将winxos在arduino上的按键模式移植到C52单片机上，调试通过。

**(1)按键与函数绑定**

**(2)周期做抖动延时**

----

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/key/key.png" style="width:600px">

----

**1.winxos-arduino**

{% highlight c %}
/*
arduino key model
can bind D0-D13 A0-A5
winxos 2014-10-23
*/
#include <arduino.h>
#define KEYDOWN_VOL LOW

#define MAX_KEYS 20
#define TRIGGER_CYC 3
#define TRIGGER_DOWN 0
#define TRIGGER_UP 1

typedef unsigned char u8;
typedef void(*CALLBACK)();

CALLBACK cbs[MAX_KEYS]; //callback array
u8 key_buf[MAX_KEYS]; //software key buffer
long key_binded = 0; //binded bit check
long key_trig_type = 0;

void set_bit(long &data, u8 port)
{
  data |= (1 << port);
}

void clear_bit(long &data, u8 port)
{
  data &= ~(1 << port);
}

u8 get_bit(long data, u8 port)
{
  return (data & (1 << port)) > 0;
}

static void add_listener(u8 port, void (*fun)(), u8 type)
{
  if (port > MAX_KEYS)
    return;
  set_bit(key_binded, port);
  if (type == TRIGGER_UP)
    set_bit(key_trig_type, port);
  else
    clear_bit(key_trig_type, port);
  cbs[port] = fun;
  Serial.println(key_trig_type,HEX);
  Serial.println(key_binded,HEX);
}
void add_keydown_listener(u8 port, void (*fun)())
{
  add_listener(port, fun, TRIGGER_DOWN);
}
void add_keyup_listener(u8 port, void (*fun)())
{
  add_listener(port, fun, TRIGGER_UP);
}
void remove_listener(u8 port)
{
  if (port > MAX_KEYS)
    return;
  clear_bit(key_binded,port);
}
void btn_scan()
{
  for (u8 i = 0; i < MAX_KEYS; i++)
  {
    if (get_bit(key_binded,i)) //binded
    {
      if (digitalRead(i) == KEYDOWN_VOL)
      {
        if (key_buf[i] < 250)
          key_buf[i]++;
        if (key_buf[i] == TRIGGER_CYC &&
            get_bit(key_trig_type, i) == TRIGGER_DOWN)
        {
          cbs[i]();//keydown callback
        }
      }
      else
      {
        if (key_buf[i] > TRIGGER_CYC &&
            get_bit(key_trig_type, i) == TRIGGER_UP)
        {
          cbs[i]();//keyup callback
        }
        key_buf[i] = 0;
      }
    }
  }
}

{% endhighlight %}

----
{% highlight c %}
//btn test
#include "btn.h"
void btn1_pressed()
{
  Serial.println("btn1 pressed.");
}
void btn2_up()
{
  Serial.println("btn2 up.");
}
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(3,INPUT_PULLUP);
  pinMode(4,INPUT_PULLUP);
  add_keydown_listener(3, &btn1_pressed);//binded to pin 3
  add_keyup_listener(4, &btn2_up);//binded to pin 4
  delay(1000);
}
void loop() {
  // put your main code here, to run repeatedly:
  btn_scan();
  delay(10);
}
{% endhighlight %}

----

**2.单片机C52**
{% highlight c %}
/*
key.c
c52 key model
bind key and event
gump 2016-6-9
*/

#include"../Headers/Key.H"
#include <reg52.h>

#define MAX_KEYS 4  
#define TRIGGER_CYC 3
#define TRIGGER_UP 1
#define TRIGGER_DOWN 0
#define KEY_PORTS P3
#define KEYDOWN_VOL 0


CALLBACK funcs[MAX_KEYS];
u8 key_buf[MAX_KEYS];

long key_binded = 0;
long key_trig_type = 0;

void set_bit(long *dat, u8 port)
{
	*dat |= (1 << port);

}

void clear_bit(long *dat, u8 port)
{
	*dat &= ~(1 << port);
}

u8 get_bit(long dat, u8 port)
{

	return (dat & (1 << port)) > 0;
}


void add_key_listener(u8 port, CALLBACK func, u8 type)
{
	if(port > MAX_KEYS)
	{
		return;
	}
	set_bit(&key_binded, port);
	if(type == TRIGGER_UP)
	{
		set_bit(&key_trig_type, port);
	}
	else
	{
		clear_bit(&key_trig_type, port);
	}
	funcs[port] = func;

}

void add_keydown_listener(u8 port, CALLBACK func)
{
    add_key_listener(port, func, TRIGGER_DOWN);
}

void add_keyup_listener(u8 port, CALLBACK func)
{
    add_key_listener(port, func, TRIGGER_UP);
}

void remove_listener(u8 port)
{
	if(port > MAX_KEYS)
	{
		return;
	}
	clear_bit(&key_binded, port);
}

bit get_bit_vol(u8 i)
{
	if(i > 7)
	{
		return 0;
	}

	return (KEY_PORTS >> i) & 0x01 == 1? 1 : 0;

}
void key_scan()
{
    u8 i = 0;
    for(i = 0; i < MAX_KEYS; ++i)
    {
        if(get_bit(key_binded, i))
        {
            if(get_bit_vol(i) == KEYDOWN_VOL)
            {
                if(key_buf[i] < 250)
                {
                    ++key_buf[i];
                }
                if(key_buf[i] == TRIGGER_CYC &&
                        get_bit(key_trig_type, i) == TRIGGER_DOWN)
                {
                    funcs[i]();
                }
            }
            else
            {
                if(key_buf[i] > TRIGGER_CYC &&
                        get_bit(key_trig_type, i) == TRIGGER_UP)
                {
                    funcs[i]();
                }
                key_buf[i] = 0;
            }
        }
    }
}

{% endhighlight %}
----

{% highlight c %}
/*key.h*/
#ifndef __KEY_H__
#define __KEY_H__
typedef unsigned char u8;
typedef void (*CALLBACK)(void);

void add_keydown_listener(u8 port, CALLBACK func);
void add_keyup_listener(u8 port, CALLBACK func);
void key_scan();

#endif
{% endhighlight %}
---
