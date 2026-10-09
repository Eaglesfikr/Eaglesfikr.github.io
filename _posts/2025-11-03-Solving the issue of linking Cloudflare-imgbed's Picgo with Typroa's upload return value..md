---
title: Solving the issue of linking Cloudflare-imgbed's Picgo with Typroa's upload return value.
description: The official Cloudflare-imgbed documentation provides configuration methods in its Q&A section,but it has some questions.
author: eaglesfikr
date: 2025-11-03 11:33:00 +0800
categories: [my blog]
tags: [imgbed]
pin: true
math: true
mermaid: true
image:
  path: /assets/commons/banner.png
  lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
  alt: a sloving of imgbed with picgo.

---

**The official Cloudflare-imgbed documentation provides configuration methods in its Q&A section**

>[https://cfbed.sanyue.de/qa/](https://cfbed.sanyue.de/qa/)
{: .prompt-tip }

**However, under normal circumstances, even if the mirror address is changed, it's impossible to directly install its customizable prefix web-uploader plugin in Picgo. Therefore, we searched for resources elsewhere:**

>[Awesome-PicGo/README.md at master · PicGo/Awesome-PicGo](https://github.com/PicGo/Awesome-PicGo/blob/master/README.md)
{: .prompt-tip }

**The aforementioned repository provides a custom `picgo-plugin-custom-api-uploader` plugin. We select it and modify the configuration according to our needs.**

## 1.**Import Local Files**

In PicGo, select Import Local Files：

![](https://imgbed.7ingwe1.top/file/1762143192933_asdfgweeow.jpg)

Select the folder containing the downloaded repository.

## 2.imgbed **configuration**

Select the image hosting settings and choose the custom interface to upload，Fill in the content as shown in the picture：

![](https://imgbed.7ingwe1.top/file/1762170495500_qwerttrt.jpg)

The 3 required items are the same as in the official documentation, but the JSON path here has been modified by the author to be the return value path. The 0.src here refers to result[0].src, so it is the domain name.If you want to use it together with Typora, this is required; otherwise, the returned path after uploading the image will be incorrect and will lack the domain prefix.