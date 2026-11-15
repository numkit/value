// include/numkit/value/error.hpp
//
// numkit::Error — the runtime error type carrying an optional source
// location (line/col/function/context) and a diagnostic identifier.
#pragma once

#include <keel/format.h>
#include <keel/string.h>

#include <stdexcept>

namespace numkit {

class Error : public std::runtime_error
{
public:
    explicit Error(keel::StringRef msg,
                   keel::i32 line = 0,
                   keel::i32 col = 0,
                   keel::StringRef funcName = "",
                   keel::StringRef context = "",
                   keel::StringRef identifier = "")
        : std::runtime_error(keel::SmallString<256>(msg).c_str())
        , line_(line)
        , col_(col)
        , funcName_(funcName)
        , context_(context)
        , identifier_(identifier)
    {}

    Error(const Error &other)
        : std::runtime_error(other)
        , line_(other.line_)
        , col_(other.col_)
        , funcName_(other.funcName_.clone())
        , context_(other.context_.clone())
        , identifier_(other.identifier_.clone())
    {}

    Error &operator=(const Error &other)
    {
        if (this != &other) {
            std::runtime_error::operator=(other);
            line_ = other.line_;
            col_ = other.col_;
            funcName_ = other.funcName_.clone();
            context_ = other.context_.clone();
            identifier_ = other.identifier_.clone();
        }
        return *this;
    }

    Error(Error &&other) noexcept = default;
    Error &operator=(Error &&other) noexcept = default;
    ~Error() noexcept override = default;

    keel::i32 line() const noexcept { return line_; }
    keel::i32 col() const noexcept { return col_; }
    const keel::String &funcName() const noexcept { return funcName_; }
    const keel::String &context() const noexcept { return context_; }
    const keel::String &identifier() const noexcept { return identifier_; }

    void attachIfMissing(keel::i32 line,
                         keel::i32 col,
                         keel::StringRef funcName = "",
                         keel::StringRef context = "")
    {
        if (line_ == 0)
            line_ = line;
        if (col_ == 0)
            col_ = col;
        if (funcName_.empty() && !funcName.empty())
            funcName_.assign(funcName);
        if (context_.empty() && !context.empty())
            context_.assign(context);
    }

    keel::String formattedWhat() const
    {
        if (line_ <= 0)
            return keel::String(what());
        keel::String result = keel::format("Error at line {}", line_);
        if (col_ > 0)
            keel::format_into(result, ", column {}", col_);
        if (!funcName_.empty())
            keel::format_into(result, " in '{}'", funcName_);
        result.append(":\n  ");
        result.append(what());
        if (!context_.empty())
            keel::format_into(result, " ({})", context_);
        return result;
    }

private:
    keel::i32 line_;
    keel::i32 col_;
    keel::String funcName_;
    keel::String context_;
    keel::String identifier_;
};

} // namespace numkit
