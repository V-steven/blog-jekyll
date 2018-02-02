(function() {
    var m = function(j) { if (j.length <= 1) return null; for (var f = [], h = 1; h < j.length; h++) f.push(j[h]); return f },
        n = function(j, f) { for (var h = [], k = 0; k < j.length; k++) h.push(j[k]); for (; h.length < f;) h.push(void 0); return h },
        l = function(j) {
            if (!j.modules["async-powerpack"]) {
                if (!j.modules.async) throw Error('Missing essential components, please initialize "jscex-async" module first.');
                var f = j.Async,
                    h = f.Task,
                    k = f.CanceledError;
                f.sleep = function(a, b) {
                    return h.create(function(c) {
                        b && b.isCancellationRequested && c.complete("failure",
                            new k);
                        var e, d;
                        b && (d = function() { clearTimeout(e);
                            c.complete("failure", new k) });
                        e = setTimeout(function() { b && b.unregister(d);
                            c.complete("success") }, a);
                        b && b.register(d)
                    })
                };
                f.onEvent = function(a, b, c) {
                    return h.create(function(e) {
                        c && c.isCancellationRequested && e.complete("failure", new k);
                        var d = function() { a.removeEventListener ? a.removeEventListener(b, g) : a.removeListener ? a.removeListener(b, g) : a.detachEvent(b, g) },
                            g, i;
                        c && (i = function() { d();
                            e.complete("failure", new k) });
                        g = function(a) {
                            c && c.unregister(i);
                            d();
                            e.complete("success",
                                a)
                        };
                        a.addEventListener ? a.addEventListener(b, g) : a.addListener ? a.addListener(b, g) : a.attachEvent(b, g);
                        c && c.register(i)
                    })
                };
                h.whenAll = function() {
                    var a = {},
                        b;
                    if (arguments.length == 1) { var c = arguments[0];
                        h.isTask(c) ? (a[0] = c, b = !0) : (a = c, b = Object.prototype.toString.call(a) === "[object Array]") } else { for (c = 0; c < arguments.length; c++) a[c] = arguments[c];
                        b = !0 }
                    return h.create(function(e) {
                        var d = {},
                            g;
                        for (g in a)
                            if (a.hasOwnProperty(g)) { var i = a[g];
                                h.isTask(i) && (d[i.id] = g) }
                        for (var c in d) i = a[d[c]], i.status == "ready" && i.start();
                        for (c in d)
                            if (i = a[d[c]], i.error) { e.complete("failure", i.error); return }
                        var f = b ? [] : {},
                            j = function(b) { if (b.error) { for (var c in d) a[d[c]].removeEventListener("complete", j);
                                    e.complete("failure", b.error) } else f[d[b.id]] = b.result, delete d[b.id], k--, k == 0 && e.complete("success", f) },
                            k = 0;
                        for (c in d) g = d[c], i = a[g], i.status == "succeeded" ? (f[g] = i.result, delete d[i.id]) : (k++, i.addEventListener("complete", j));
                        k == 0 && e.complete("success", f)
                    })
                };
                h.whenAny = function() {
                    var a = {};
                    if (arguments.length == 1) {
                        var b = arguments[0];
                        h.isTask(b) ?
                            a[0] = b : a = b
                    } else
                        for (b = 0; b < arguments.length; b++) a[b] = arguments[b];
                    return h.create(function(b) { var e = {},
                            d; for (d in a)
                            if (a.hasOwnProperty(d)) { var g = a[d];
                                h.isTask(g) && (e[g.id] = d) }
                        for (var i in e) g = a[e[i]], g.status == "ready" && g.start(); for (i in e)
                            if (d = e[i], g = a[d], g.error || g.status == "succeeded") { b.complete("success", { key: d, task: g }); return }
                        var f = function(d) { for (var g in e) a[e[g]].removeEventListener("complete", f);
                            b.complete("success", { key: e[d.id], task: d }) }; for (i in e) a[e[i]].addEventListener("complete", f) })
                };
                if (!f.Jscexify) f.Jscexify = {};
                f = f.Jscexify;
                f.fromStandard = function(a) { var b = m(arguments); return function() { var c = this,
                            e = n(arguments, a.length - 1); return h.create(function(d) { e.push(function(a, c) { if (a) d.complete("failure", a);
                                else if (b) { for (var e = {}, f = 0; f < b.length; f++) e[b[f]] = arguments[f + 1]; return d.complete("success", e) } else d.complete("success", c) });
                            a.apply(c, e) }) } };
                f.fromCallback = function(a) {
                    var b = m(arguments);
                    return function() {
                        var c = this,
                            e = n(arguments, a.length - 1);
                        return h.create(function(d) {
                            e.push(function(a) {
                                if (b) {
                                    for (var c = {}, e = 0; e < b.length; e++) c[b[e]] = arguments[e];
                                    d.complete("success", c)
                                } else d.complete("success", a)
                            });
                            a.apply(c, e)
                        })
                    }
                };
                j.modules["async-powerpack"] = !0
            }
        },
        o = typeof define === "function" && !define.amd,
        p = typeof require === "function" && typeof define === "function" && define.amd;
    if (typeof require === "function" && typeof module !== "undefined" && module.exports) module.exports.init = l;
    else if (o) define("jscex-async-powerpack", ["jscex-async"], function(j, f, h) { h.exports.init = l });
    else if (p) define("jscex-async-powerpack", ["jscex-async"],
        function() { return { init: l } });
    else { if (typeof Jscex === "undefined") throw Error('Missing the root object, please load "jscex" module first.');
        l(Jscex) }
})();