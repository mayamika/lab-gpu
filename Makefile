# packing

OUTPUT_DIR := ./packages/
PACKAGES := $(addprefix $(OUTPUT_DIR),$(addsuffix .tar,$(wildcard lab*)))
SIGNER_EMAIL= andranik.chakiryan@gmail.com

.PHONY: $(PACKAGES)
$(PACKAGES): $(addprefix $(OUTPUT_DIR),%.tar):
	@mkdir -p $(OUTPUT_DIR)
	$(MAKE) -C "$*" clean
	tar -cf "$@" "$*"
	$(RM) "$@.asc"
	gpg -u $(SIGNER_EMAIL) -ab "$@"

all: $(PACKAGES)

.PHONY: clean
clean:
	$(RM) -r $(OUTPUT_DIR)