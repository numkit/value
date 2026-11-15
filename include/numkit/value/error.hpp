// include/numkit/value/error.hpp
//
// numkit::Error — the runtime error type carrying an optional source
// location (line/col/function/context) and a diagnostic identifier.
#pragma once

#include <keel/core.h>

#include <stdexcept>
#include <string>

namespace numkit {

class Error : public std::runtime_error
{
public:
    Error(const std::string &msg,
          keel::i32 line = 0,
          keel::i32 col = 0,
          const std::string &funcName = "",
          const std::string &context = "",
          const std::string &identifier = "")
        : std::runtime_error(msg)
        , line_(line)
        , col_(col)
        , funcName_(funcName)
        , context_(context)
        , identifier_(identifier)
    {}

    keel::i32 line() const noexcept { return line_; }
    keel::i32 col() const noexcept { return col_; }
    const std::string &funcName() const noexcept { return funcName_; }
    const std::string &context() const noexcept { return context_; }
    const std::string &identifier() const noexcept { return identifier_; }

    void attachIfMissing(keel::i32 line,
                         keel::i32 col,
                         const std::string &funcName = "",
                         const std::string &context = "")
    {
        if (line_ == 0)
            line_ = line;
        if (col_ == 0)
            col_ = col;
        if (funcName_.empty() && !funcName.empty())
            funcName_ = funcName;
        if (context_.empty() && !context.empty())
            context_ = context;
    }

    std::string formattedWhat() const
    {
        if (line_ <= 0)
            return what();
        std::string result = "Error at line " + std::to_string(line_);
        if (col_ > 0)
            result += ", column " + std::to_string(col_);
        if (!funcName_.empty())
            result += " in '" + funcName_ + "'";
        result += ":\n  ";
        result += what();
        if (!context_.empty())
            result += " (" + context_ + ")";
        return result;
    }

private:
    keel::i32 line_;
    keel::i32 col_;
    std::string funcName_;
    std::string context_;
    std::string identifier_;
};

} // namespace numkit
