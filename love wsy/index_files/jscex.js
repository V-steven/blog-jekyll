(function() {
    var b = { DEBUG: 1, INFO: 2, WARN: 3, ERROR: 4 },
        d = function() { this.level = b.WARN };
    d.prototype = { log: function(a) { try { console.log(a) } catch (b) {} }, debug: function(a) { this.level <= b.DEBUG && this.log(a) }, info: function(a) { this.level <= b.INFO && this.log(a) }, warn: function(a) { this.level <= b.WARN && this.log(a) }, error: function(a) { this.level <= b.ERROR && this.log(a) } };
    var e = function(a) { var b = [],
                c; for (c in a) b.push(c); return b },
        c = function(a) {
            a._forInKeys = e;
            a.Logging = { Logger: d, Level: b };
            a.logger = new d;
            a.modules = {};
            a.binders = {};
            a.builders = {}
        },
        f = typeof define === "function" && !define.amd,
        g = typeof require === "function" && typeof define === "function" && define.amd;
    typeof require === "function" && typeof module !== "undefined" && module.exports ? c(module.exports) : f ? define("jscex", function(a, b, d) { c(d.exports) }) : g ? define("jscex", function() { var a = {};
        c(a); return a }) : (typeof Jscex == "undefined" && (Jscex = {}), c(Jscex))
})();