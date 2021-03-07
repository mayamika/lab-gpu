# Packing and signing

OUTPUT_DIR ?= ./packages/
SIGNER_EMAIL ?= andranik.chakiryan@gmail.com

PACKAGES := $(addprefix $(OUTPUT_DIR)/,$(addsuffix .tar,$(wildcard lab*)))
SIGNS := $(addsuffix .asc,$(PACKAGES))

all: $(PACKAGES) $(SIGNS)

$(OUTPUT_DIR):
	mkdir $(OUTPUT_DIR)

.PHONY: $(PACKAGES)
$(PACKAGES): $(OUTPUT_DIR)/%.tar: | $(OUTPUT_DIR)
	$(MAKE) -C "$*" clean
	tar -cf "$@" "$*"

%.asc: %
	$(RM) $@
	gpg -u $(SIGNER_EMAIL) -ab "$*"

.PHONY: clean
clean:
	$(RM) -r $(OUTPUT_DIR)
